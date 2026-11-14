import os

import torch
from torch.utils.cpp_extension import load

# 3080 Ti 是 sm_86
CUDA_FLAGS = [
    "-O3",
    "-gencode=arch=compute_86,code=sm_86",
]


def build(name, src_dir, sources):
    paths = [os.path.join(src_dir, s) for s in sources]
    return load(
        name=name,
        sources=paths,
        extra_cuda_cflags=CUDA_FLAGS,
        verbose=False,
    )


def cuda_bench(fn, warmup=10, iters=100):
    # 返回平均耗时 ms
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def check_close(out, ref, atol=1e-2, rtol=1e-3, name=""):
    diff = (out.float() - ref.float()).abs()
    ok = torch.allclose(out.float(), ref.float(), atol=atol, rtol=rtol)
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name} max_abs_err={diff.max().item():.2e}")
    return ok
