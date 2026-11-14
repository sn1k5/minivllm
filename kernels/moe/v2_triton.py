import torch
import triton
import triton.language as tl

from gemm.triton_gemm import gemm as triton_gemm


# 每token一个program, 做softmax+topk+renorm
@triton.jit
def route_topk_kernel(gating_ptr, ids_ptr, weights_ptr,
                      T, E, K: tl.constexpr, BLOCK_E: tl.constexpr):
    pid = tl.program_id(0)
    offs_e = tl.arange(0, BLOCK_E)
    mask = offs_e < E
    g = tl.load(gating_ptr + pid * E + offs_e, mask=mask, other=-float("inf"))
    mx = tl.max(g, axis=0)
    p = tl.exp(g - mx)
    s = tl.sum(p, axis=0)
    p = p / s
    # 暴力取topk, E不大
    ids = tl.zeros((K,), dtype=tl.int32)
    wts = tl.zeros((K,), dtype=tl.float32)
    total = 0.0
    for k in range(K):
        best = tl.argmax(p, axis=0)
        val = tl.max(p, axis=0)
        ids = tl.where(tl.arange(0, K) == k, best, ids)
        wts = tl.where(tl.arange(0, K) == k, val, wts)
        total += val
        p = tl.where(offs_e == best, -1.0, p)
    wts = wts / total
    tl.store(ids_ptr + pid * K + tl.arange(0, K), ids)
    tl.store(weights_ptr + pid * K + tl.arange(0, K), wts)


# 用torch做计数和前缀和, 简单可靠
def _count_and_sort(ids, E, topk):
    T = ids.shape[0]
    # ids: [T, K] -> [T*K]
    flat = ids.reshape(-1)
    counts = torch.zeros(E, dtype=torch.int32, device=ids.device)
    counts.scatter_add_(0, flat, torch.ones_like(flat))
    poff = torch.zeros(E + 1, dtype=torch.int32, device=ids.device)
    poff[1:] = torch.cumsum(counts, dim=0)
    # 按专家分组排序token
    order = flat.argsort(stable=True)
    sorted_tokens = order
    return counts, poff, sorted_tokens


def moe_triton(hidden, w13, w2, gating, topk):
    T, H = hidden.shape
    E = gating.shape[1]
    I = w2.shape[2]
    K = topk

    BLOCK_E = triton.next_power_of_2(E)
    ids = torch.empty(T, K, dtype=torch.int32, device=hidden.device)
    weights = torch.empty(T, K, dtype=torch.float32, device=hidden.device)
    route_topk_kernel[(T,)](gating, ids, weights, T, E, K, BLOCK_E)

    counts, poff, sorted_tokens = _count_and_sort(ids, E, K)
    total = T * K
    # 排序后的hidden行
    token_idx = sorted_tokens // K
    x_sorted = hidden[token_idx]  # [total, H]

    # 第一次GEMM: x @ w13^T -> [total, 2I]
    # 分组做: 每个专家一段
    inter = torch.empty(total, 2 * I, device=hidden.device, dtype=hidden.dtype)
    for e in range(E):
        s = poff[e].item()
        cnt = counts[e].item()
        if cnt == 0:
            continue
        x = x_sorted[s:s + cnt]
        w = w13[e]  # [2I, H]
        inter[s:s + cnt] = (x @ w.t()).half()

    # silu(gate) * up
    g = inter[:, :I]
    u = inter[:, I:]
    act = torch.nn.functional.silu(g.float()) * u.float()
    act = act.half()

    # 第二次GEMM: act @ w2^T -> [total, H], 按token加权
    out = torch.zeros(T, H, dtype=torch.float32, device=hidden.device)
    for e in range(E):
        s = poff[e].item()
        cnt = counts[e].item()
        if cnt == 0:
            continue
        a = act[s:s + cnt]
        w = w2[e]  # [H, I]
        y = a @ w.t()  # [cnt, H]
        # 对应权重
        wgt = weights.view(-1)[sorted_tokens[s:s + cnt]].float()
        out.index_add_(0, token_idx[s:s + cnt], y.float() * wgt[:, None])

    return out.half()
