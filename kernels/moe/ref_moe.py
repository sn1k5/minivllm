import torch
import torch.nn.functional as F


def moe_ref(hidden, w13, w2, gating, topk):
    # fp32参考实现, 对齐Qwen3-MoE: softmax->topk->renorm
    T, H = hidden.shape
    E = gating.shape[1]
    I = w2.shape[2]
    prob = torch.softmax(gating.float(), dim=-1)
    w, eid = torch.topk(prob, topk, dim=-1)
    w = w / w.sum(-1, keepdim=True)
    out = torch.zeros(T, H, dtype=torch.float32, device=hidden.device)
    for e in range(E):
        mask = eid == e
        if not mask.any():
            continue
        rows, ks = mask.nonzero(as_tuple=True)
        x = hidden[rows].float()
        h = x @ w13[e].float().t()
        g, u = h[:, :I], h[:, I:]
        act = F.silu(g) * u
        y = act @ w2[e].float().t()
        out.index_add_(0, rows, y * w[rows, ks].unsqueeze(1))
    return out


def moe_v0(hidden, w13, w2, gating, topk):
    # v0 baseline: 逐专家mask循环, 全是kernel launch
    T, H = hidden.shape
    E = gating.shape[1]
    I = w2.shape[2]
    prob = torch.softmax(gating.float(), dim=-1)
    w, eid = torch.topk(prob, topk, dim=-1)
    w = w / w.sum(-1, keepdim=True)
    out = torch.zeros(T, H, dtype=torch.float32, device=hidden.device)
    for e in range(E):
        mask = eid == e
        if not mask.any():
            continue
        rows, ks = mask.nonzero(as_tuple=True)
        x = hidden[rows].half()
        h = x @ w13[e].t()
        g, u = h[:, :I], h[:, I:]
        act = F.silu(g) * u
        y = act @ w2[e].t()
        out.index_add_(0, rows, y.float() * w[rows, ks].unsqueeze(1))
    return out.half()
