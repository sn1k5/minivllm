import os
import sys

import torch

from common import build, cuda_bench, check_close

sys.path.append(os.path.join(os.path.dirname(__file__), "moe"))
sys.path.append(os.path.join(os.path.dirname(__file__), "moe", "vllm_ref"))
from moe.ref_moe import moe_ref, moe_v0  # noqa: E402
from moe.v2_triton import moe_triton  # noqa: E402
from vllm_fused_moe import fused_experts  # noqa: E402

torch.manual_seed(0)
E, TOPK, H, I = 128, 8, 2048, 768
TS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]


def make_data(T):
    hidden = torch.randn(T, H, device="cuda", dtype=torch.float16)
    w13 = torch.randn(E, 2 * I, H, device="cuda", dtype=torch.float16) * 0.02
    w2 = torch.randn(E, H, I, device="cuda", dtype=torch.float16) * 0.02
    gating = torch.randn(T, E, device="cuda", dtype=torch.float32)
    return hidden, w13, w2, gating


def topk_torch(gating, K):
    prob = torch.softmax(gating.float(), dim=-1)
    w, ids = torch.topk(prob, K, dim=-1)
    w = w / w.sum(-1, keepdim=True)
    return w.contiguous(), ids.to(torch.int32).contiguous()


def peak_mem(fn):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 1024 ** 2


def gflops(T, ms):
    # w13: 2*2I*H, w2: 2*H*I, 共6IH每槽位
    return 6.0 * T * TOPK * I * H / (ms / 1000.0) / 1e12


def bench_all():
    v1 = build("v1_moe", "moe", ["v1_cuda.cu"])
    v2 = build("v2_moe", "moe", ["v2_cuda.cu"])
    print("=== 正确性 (T=64) ===")
    hidden, w13, w2, gating = make_data(64)
    ref = moe_ref(hidden, w13, w2, gating, TOPK)
    out = torch.empty(64, H, device="cuda", dtype=torch.float16)
    v1.moe_forward(hidden, w13, w2, gating, TOPK, out)
    check_close(out, ref, atol=0.1, rtol=1e-2, name="v1_cuda")
    v2.moe_forward(hidden, w13, w2, gating, TOPK, out)
    check_close(out, ref, atol=0.1, rtol=1e-2, name="v2_cuda")
    check_close(moe_triton(hidden, w13, w2, gating, TOPK), ref,
                atol=0.1, rtol=1e-2, name="v2_triton")
    check_close(moe_v0(hidden, w13, w2, gating, TOPK), ref,
                atol=0.1, rtol=1e-2, name="v0_torch")
    w_t, ids_t = topk_torch(gating, TOPK)
    check_close(fused_experts(hidden, w13, w2, w_t, ids_t), ref,
                atol=0.1, rtol=1e-2, name="vllm_fused")

    print("\n=== 延迟扫描 us (vllm用默认config) ===")
    hdr = ["T", "v0_torch", "v1_cuda", "v2_cuda", "v2_triton", "vllm",
           "v1/v0", "vllm/v1", "TFLOPS_v1", "TFLOPS_vllm"]
    print(" | ".join(f"{h:>10}" for h in hdr))
    results = {}
    for T in TS:
        hidden, w13, w2, gating = make_data(T)
        out = torch.empty(T, H, device="cuda", dtype=torch.float16)
        w_t, ids_t = topk_torch(gating, TOPK)
        iters = 100 if T <= 256 else (50 if T <= 1024 else 20)
        t0 = cuda_bench(lambda: moe_v0(hidden, w13, w2, gating, TOPK),
                        iters=iters)
        t1 = cuda_bench(
            lambda: v1.moe_forward(hidden, w13, w2, gating, TOPK, out),
            iters=iters)
        t2 = cuda_bench(
            lambda: v2.moe_forward(hidden, w13, w2, gating, TOPK, out),
            iters=iters)
        t3 = cuda_bench(lambda: moe_triton(hidden, w13, w2, gating, TOPK),
                        iters=iters)
        t4 = cuda_bench(lambda: fused_experts(hidden, w13, w2, w_t, ids_t),
                        iters=iters)
        row = [str(T), f"{t0*1000:.1f}", f"{t1*1000:.1f}", f"{t2*1000:.1f}",
               f"{t3*1000:.1f}", f"{t4*1000:.1f}", f"{t0/t1:.1f}x",
               f"{t1/t4:.2f}x", f"{gflops(T, t1):.1f}", f"{gflops(T, t4):.1f}"]
        print(" | ".join(f"{c:>10}" for c in row))
        results[T] = (t0, t1, t2, t3, t4)
    return results


def tune_vllm():
    # 按decode/prefill两档搜最优config
    decode_cfgs = [
        {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64,
         "GROUP_SIZE_M": 1},
        {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64,
         "GROUP_SIZE_M": 1, "num_warps": 4},
        {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32,
         "GROUP_SIZE_M": 1, "num_warps": 4},
        {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32,
         "GROUP_SIZE_M": 1, "num_warps": 4},
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32,
         "GROUP_SIZE_M": 8, "num_warps": 4},
    ]
    prefill_cfgs = [
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32,
         "GROUP_SIZE_M": 8, "num_warps": 4},
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32,
         "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 4},
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32,
         "GROUP_SIZE_M": 8, "num_warps": 8, "num_stages": 3},
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64,
         "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 4},
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64,
         "GROUP_SIZE_M": 8, "num_warps": 8},
    ]
    print("\n=== vllm config搜索 ===")
    best = {}
    for T, cfgs in [(1, decode_cfgs), (8, decode_cfgs), (64, decode_cfgs),
                    (512, prefill_cfgs), (4096, prefill_cfgs)]:
        hidden, w13, w2, gating = make_data(T)
        w_t, ids_t = topk_torch(gating, TOPK)
        for i, cfg in enumerate(cfgs):
            try:
                t = cuda_bench(
                    lambda: fused_experts(hidden, w13, w2, w_t, ids_t,
                                          override_config=cfg), iters=50)
                print(f"T={T:<5} cfg{i} BSM={cfg['BLOCK_SIZE_M']:>3} "
                      f"BSN={cfg['BLOCK_SIZE_N']:>3} BSK={cfg['BLOCK_SIZE_K']:>2} "
                      f"-> {t*1000:8.1f}us")
                if T not in best or t < best[T][0]:
                    best[T] = (t, cfg)
            except Exception as ex:
                print(f"T={T} cfg{i} failed: {type(ex).__name__}")
    for T, (t, cfg) in best.items():
        print(f"best T={T}: {t*1000:.1f}us {cfg}")
    return best


def mem_bench():
    v1 = build("v1_moe", "moe", ["v1_cuda.cu"])
    v2 = build("v2_moe", "moe", ["v2_cuda.cu"])
    print("\n=== 激活峰值显存 MB ===")
    for T in [64, 1024]:
        hidden, w13, w2, gating = make_data(T)
        out = torch.empty(T, H, device="cuda", dtype=torch.float16)
        w_t, ids_t = topk_torch(gating, TOPK)
        m_v0 = peak_mem(lambda: moe_v0(hidden, w13, w2, gating, TOPK))
        m_v1 = peak_mem(
            lambda: v1.moe_forward(hidden, w13, w2, gating, TOPK, out))
        m_vl = peak_mem(lambda: fused_experts(hidden, w13, w2, w_t, ids_t))
        print(f"T={T:<5} v0_torch {m_v0:8.1f} | v1_cuda {m_v1:8.1f} | "
              f"vllm {m_vl:8.1f}")


def stage_breakdown():
    v1 = build("v1_moe", "moe", ["v1_cuda.cu"])
    from torch.profiler import profile, ProfilerActivity
    print("\n=== v1_cuda 分阶段 (20次平均, us) ===")
    for T in [1, 512]:
        hidden, w13, w2, gating = make_data(T)
        out = torch.empty(T, H, device="cuda", dtype=torch.float16)
        for _ in range(10):
            v1.moe_forward(hidden, w13, w2, gating, TOPK, out)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(20):
                v1.moe_forward(hidden, w13, w2, gating, TOPK, out)
            torch.cuda.synchronize()
        print(f"--- T={T} ---")
        ka = prof.key_averages()
        total = 0.0
        stages = {}
        for ev in ka:
            if ev.self_device_time_total > 0 and ev.count >= 20:
                us = ev.self_device_time_total / 20.0
                name = ev.key.split("(")[0][:40]
                stages[name] = us
                total += us
        for name, us in sorted(stages.items(), key=lambda x: -x[1]):
            print(f"  {name:<40} {us:8.1f}us ({us/total*100:4.1f}%)")
        print(f"  {'总计':<40} {total:8.1f}us")


if __name__ == "__main__":
    bench_all()
    tune_vllm()
    mem_bench()
    stage_breakdown()
