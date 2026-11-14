import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "moe"))

from common import build, check_close  # noqa: E402
from moe.ref_moe import moe_ref, moe_v0  # noqa: E402

torch.manual_seed(0)

# Qwen3-30B-A3B单层的MoE形状
E, TOPK = 128, 8
H, I = 2048, 768


def make_data(T):
    hidden = torch.randn(T, H, device="cuda", dtype=torch.float16)
    w13 = torch.randn(E, 2 * I, H, device="cuda", dtype=torch.float16) * 0.02
    w2 = torch.randn(E, H, I, device="cuda", dtype=torch.float16) * 0.02
    gating = torch.randn(T, E, device="cuda", dtype=torch.float32)
    return hidden, w13, w2, gating


def main():
    mod = build("v1_moe", "moe", ["v1_cuda.cu"])
    print("v1_moe build ok")

    for T in [1, 7, 64, 512]:
        hidden, w13, w2, gating = make_data(T)
        ref = moe_ref(hidden, w13, w2, gating, TOPK)
        out = torch.empty(T, H, device="cuda", dtype=torch.float16)
        mod.moe_forward(hidden, w13, w2, gating, TOPK, out)
        check_close(out, ref, atol=0.1, rtol=1e-2, name=f"v1_cuda T={T}")

    # v0 baseline也顺带验证下口径一致
    T = 64
    hidden, w13, w2, gating = make_data(T)
    ref = moe_ref(hidden, w13, w2, gating, TOPK)
    v0 = moe_v0(hidden, w13, w2, gating, TOPK)
    check_close(v0, ref, atol=0.1, rtol=1e-2, name="v0_torch T=64")


if __name__ == "__main__":
    main()
