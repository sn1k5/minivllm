import torch
import triton
import triton.language as tl


def get_configs():
    # 几组tile配置让autotune挑, 覆盖大小M
    cfgs = [
        (128, 128, 32, 8, 8, 3),
        (128, 64, 32, 8, 4, 3),
        (64, 128, 32, 8, 8, 3),
        (64, 64, 32, 4, 4, 3),
        (64, 32, 32, 4, 4, 3),
        (32, 64, 32, 4, 4, 3),
        (32, 32, 32, 4, 4, 3),
        (16, 128, 64, 1, 4, 3),
        (16, 64, 64, 1, 2, 3),
        (16, 32, 64, 1, 2, 3),
    ]
    return [
        triton.Config({"BM": bm, "BN": bn, "BK": bk, "GM": gm},
                      num_warps=w, num_stages=s)
        for bm, bn, bk, gm, w, s in cfgs
    ]


@triton.autotune(configs=get_configs(), key=["M", "N", "K"])
@triton.jit
def gemm_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                sam, sak, sbk, sbn, scm, scn,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                GM: tl.constexpr):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BM)
    num_n = tl.cdiv(N, BN)
    # 分组排序让同组block共用A, 提高L2命中
    group_id = pid // (GM * num_n)
    first_m = group_id * GM
    gsize = min(num_m - first_m, GM)
    bid_m = first_m + (pid % gsize)
    bid_n = (pid % (GM * num_n)) // gsize

    rm = bid_m * BM + tl.arange(0, BM)
    rn = bid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    pa = a_ptr + rm[:, None] * sam + rk[None, :] * sak
    pb = b_ptr + rk[:, None] * sbk + rn[None, :] * sbn

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    mm = rm[:, None] < M
    mn = rn[None, :] < N
    for k in range(0, tl.cdiv(K, BK)):
        km = k * BK
        a = tl.load(pa, mask=mm & (rk[None, :] + km < K), other=0.0)
        b = tl.load(pb, mask=(rk[:, None] + km < K) & mn, other=0.0)
        acc = tl.dot(a, b, acc)
        pa += BK * sak
        pb += BK * sbk

    pc = c_ptr + rm[:, None] * scm + rn[None, :] * scn
    tl.store(pc, acc.to(tl.float16), mask=mm & mn)


def gemm(A, B):
    # fp16输入, fp16输出
    M, K = A.shape
    N = B.shape[1]
    C = torch.empty(M, N, device=A.device, dtype=torch.float16)
    grid = lambda meta: (triton.cdiv(M, meta["BM"]) * triton.cdiv(N, meta["BN"]),)
    gemm_kernel[grid](A, B, C, M, N, K,
                      A.stride(0), A.stride(1), B.stride(0), B.stride(1),
                      C.stride(0), C.stride(1))
    return C
