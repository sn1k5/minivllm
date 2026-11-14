import os
import sys

import torch

from common import build, cuda_bench

sys.path.append(os.path.join(os.path.dirname(__file__), "gemm"))
import triton_gemm  # noqa: E402

torch.manual_seed(0)

# Qwen3-0.6B的典型GEMM形状
SHAPES = [
    ("qkv", 1024, 4096),
    ("gate_up", 1024, 6144),
    ("down", 3072, 1024),
    ("o_proj", 1024, 1024),
]
MS_DECODE = [1, 8, 32, 64]
MS_PREFILL = [1024, 4096]


def gflops(M, N, K, ms):
    return 2.0 * M * N * K / (ms / 1000.0) / 1e9


def bench_fp32():
    mods = {v: build(v, "gemm", [f"{v}.cu"]) for v in
            ["v0_naive", "v1_smem", "v2_regtile", "v3_double_buf"]}
    print("=== fp32 vs cuBLAS(无TF32) ===")
    for tag, K, N in SHAPES:
        for M in MS_DECODE + MS_PREFILL:
            A = torch.randn(M, K, device="cuda")
            B = torch.randn(K, N, device="cuda")
            t_cublas = cuda_bench(lambda: torch.matmul(A, B))
            row = [f"{tag} M={M:<5}", f"cublas {t_cublas*1000:7.1f}us"]
            for name, mod in mods.items():
                C = torch.empty(M, N, device="cuda")
                if name == "v3_double_buf" and (M % 128 or N % 128 or K % 8):
                    continue  # 尺寸不齐跳过
                t = cuda_bench(lambda: mod.gemm(A, B, C))
                row.append(f"{name} {t*1000:8.1f}us ({t/t_cublas*100:5.1f}%)")
            print("  ".join(row))


def pad16(A, B, BM=64, BN=64, BK=16):
    M, K = A.shape
    N = B.shape[1]
    Mp = (M + BM - 1) // BM * BM
    Np = (N + BN - 1) // BN * BN
    Kp = (K + BK - 1) // BK * BK
    Ap = torch.zeros(Mp, Kp, device=A.device, dtype=A.dtype)
    Bp = torch.zeros(Kp, Np, device=B.device, dtype=B.dtype)
    Cp = torch.empty(Mp, Np, device=A.device, dtype=A.dtype)
    Ap[:M, :K] = A
    Bp[:K, :N] = B
    return Ap, Bp, Cp, Mp, Np


def bench_fp16():
    mod = build("v4_wmma", "gemm", ["v4_wmma.cu"])
    mod_b = build("v4b_wmma", "gemm", ["v4b_wmma_bk32.cu"])
    # 预warmup, 避免首次调用编译干扰
    A0 = torch.randn(64, 64, device="cuda", dtype=torch.float16)
    B0 = torch.randn(64, 64, device="cuda", dtype=torch.float16)
    C0 = torch.empty(64, 64, device="cuda", dtype=torch.float16)
    for _ in range(5):
        mod.gemm(A0, B0, C0)
        mod_b.gemm(A0, B0, C0)
        triton_gemm.gemm(A0, B0)
    torch.cuda.synchronize()
    print("=== fp16 vs cuBLAS ===")
    for tag, K, N in SHAPES:
        for M in MS_DECODE + MS_PREFILL:
            A = torch.randn(M, K, device="cuda", dtype=torch.float16)
            B = torch.randn(K, N, device="cuda", dtype=torch.float16)
            t_cublas = cuda_bench(lambda: torch.matmul(A, B))
            row = [f"{tag} M={M:<5}", f"cublas {t_cublas*1000:7.1f}us"]
            Ap, Bp, Cp, Mp, Np = pad16(A, B)
            t = cuda_bench(lambda: mod.gemm(Ap, Bp, Cp))
            row.append(f"v4 {t*1000:7.1f}us ({t/t_cublas*100:5.1f}%)")
            # v4b要求K%32==0
            Kp2 = (K + 31) // 32 * 32
            Ap2 = torch.zeros(Mp, Kp2, device="cuda", dtype=torch.float16)
            Ap2[:M, :K] = A
            Bp2 = torch.zeros(Kp2, Np, device="cuda", dtype=torch.float16)
            Bp2[:K, :N] = B
            Cp2 = torch.empty(Mp, Np, device="cuda", dtype=torch.float16)
            t = cuda_bench(lambda: mod_b.gemm(Ap2, Bp2, Cp2))
            row.append(f"v4b {t*1000:7.1f}us ({t/t_cublas*100:5.1f}%)")
            t = cuda_bench(lambda: triton_gemm.gemm(A, B))
            row.append(f"triton {t*1000:7.1f}us ({t/t_cublas*100:5.1f}%)")
            print("  ".join(row), f"| {gflops(M, N, K, t):.0f} GFLOPS")


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False
    bench_fp32()
    bench_fp16()
