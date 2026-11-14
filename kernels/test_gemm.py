import os
import sys

import torch

from common import build, check_close

sys.path.append(os.path.join(os.path.dirname(__file__), "gemm"))

torch.manual_seed(0)


def pad_gemm(mod, A, B, BM=128, BN=128, BK=8):
    # v3要求尺寸是tile整数倍, 不够就pad
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
    mod.gemm(Ap, Bp, Cp)
    return Cp[:M, :N]


def test_fp32():
    mods = {}
    for v in ["v0_naive", "v1_smem", "v2_regtile", "v3_double_buf"]:
        mods[v] = build(v, "gemm", [f"{v}.cu"])
        print(f"{v} build ok")

    shapes = [(256, 1024, 1024), (100, 3072, 6144), (4096, 1024, 6144)]
    for M, K, N in shapes:
        A = torch.randn(M, K, device="cuda")
        B = torch.randn(K, N, device="cuda")
        ref = A @ B
        print(f"--- fp32 M={M} K={K} N={N} ---")
        for name, mod in mods.items():
            C = torch.empty(M, N, device="cuda")
            if name == "v3_double_buf":
                out = pad_gemm(mod, A, B)
            else:
                mod.gemm(A, B, C)
                out = C
            check_close(out, ref, atol=1e-2, rtol=1e-2, name=name)


def test_fp16():
    mod = build("v4_wmma", "gemm", ["v4_wmma.cu"])
    print("v4_wmma build ok")
    import triton_gemm

    shapes = [(256, 1024, 1024), (100, 3072, 6144), (512, 3072, 1024)]
    for M, K, N in shapes:
        A = torch.randn(M, K, device="cuda", dtype=torch.float16)
        B = torch.randn(K, N, device="cuda", dtype=torch.float16)
        ref = (A @ B).float()
        print(f"--- fp16 M={M} K={K} N={N} ---")
        out4 = pad_gemm(mod, A, B, BM=64, BN=64, BK=16)
        check_close(out4, ref, atol=1e-1, rtol=1e-2, name="v4_wmma")
        outt = triton_gemm.gemm(A, B)
        check_close(outt, ref, atol=1e-1, rtol=1e-2, name="triton")


if __name__ == "__main__":
    test_fp32()
    test_fp16()
