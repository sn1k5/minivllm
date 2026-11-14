import os
import sys

import torch

from common import build, cuda_bench

sys.path.append(os.path.join(os.path.dirname(__file__), "moe"))
sys.path.append(os.path.join(os.path.dirname(__file__), "gemm"))
from moe.ref_moe import moe_v0  # noqa: E402
from moe.v2_triton import moe_triton  # noqa: E402

torch.manual_seed(0)
E, TOPK, H, I = 128, 8, 2048, 768


def make_data(T):
    hidden = torch.randn(T, H, device="cuda", dtype=torch.float16)
    w13 = torch.randn(E, 2 * I, H, device="cuda", dtype=torch.float16) * 0.02
    w2 = torch.randn(E, H, I, device="cuda", dtype=torch.float16) * 0.02
    gating = torch.randn(T, E, device="cuda", dtype=torch.float32)
    return hidden, w13, w2, gating


def main():
    mod = build("v1_moe", "moe", ["v1_cuda.cu"])
    print("=== MoE 前向测速 (Qwen3-MoE E=128 topk=8 H=2048 I=768) ===")
    for T in [1, 4, 16, 64, 256, 512, 1024]:
        hidden, w13, w2, gating = make_data(T)
        out = torch.empty(T, H, device="cuda", dtype=torch.float16)

        t0 = cuda_bench(lambda: moe_v0(hidden, w13, w2, gating, TOPK))
        t1 = cuda_bench(lambda: mod.moe_forward(hidden, w13, w2, gating, TOPK, out))
        t2 = cuda_bench(lambda: moe_triton(hidden, w13, w2, gating, TOPK))

        speedup = t0 / t1
        print(f"T={T:<5} | v0_torch {t0*1000:8.1f}us | v1_cuda {t1*1000:8.1f}us "
              f"({speedup:5.1f}x) | v2_triton {t2*1000:8.1f}us ({t0/t2:5.1f}x)")


if __name__ == "__main__":
    main()
