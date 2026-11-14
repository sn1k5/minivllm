from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..kv_ops import gather_fp8_kv, gather_kv_padded, store_kvcache

# ==============================================================================
# 注意力后端：fa2（flash_attn）/ torch（SDPA 兜底）双实现，flashinfer 仅小算子。
#
# flash_attn：prefill varlen + decode kvcache，支持 CUDA Graph（装了 wheel 才可用）。
# torch：gather 成 padded batch 后走 SDPA（enable_gqa），eager；任何环境可跑。
# 实测注记：flashinfer 0.6.13 的 paged attention wrapper 在本机（cu130 + sm86）
# 对 num_kv_heads > 1 输出错误（fa2/triton/cutlass/trtllm_gen 四种后端一致复现），
# 故不用作注意力后端；rmsnorm / silu / rope 小算子已数值验证正确，继续使用。
# ==============================================================================


def _flash_attn():
    """懒加载 flash_attn：模块可导入（CPU 单测），GPU 运行时才需要。"""
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

    return flash_attn_varlen_func, flash_attn_with_kvcache


class AttnInputs(NamedTuple):
    slot_mapping: torch.Tensor
    is_prefill: bool
    cu_seqlens_q: Optional[torch.Tensor] = None
    cu_seqlens_k: Optional[torch.Tensor] = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    block_tables: Optional[torch.Tensor] = None
    cache_seqlens: Optional[torch.Tensor] = None


@dataclass
class AttnMetadata:
    kv_cache: torch.Tensor
    kv_scale: Optional[torch.Tensor] = None  # C2 FP8：per-(layer, slot) scale


_ATTN_META: Optional[AttnMetadata] = None

_BACKEND = "torch"  # torch / fa2


def set_attention_backend(name: str) -> None:
    global _BACKEND
    _BACKEND = name


@contextmanager
def attention_context(meta: AttnMetadata):
    global _ATTN_META
    prev = _ATTN_META
    _ATTN_META = meta
    try:
        yield
    finally:
        _ATTN_META = prev


# ------------------------------------------------------------------ #
# torch 后端：padded batch + SDPA（enable_gqa）
# ------------------------------------------------------------------ #

# gather 缓冲复用：按 (bs, 对齐后的 max_kv) 缓存，padding 区被 mask 屏蔽。
_PAD_CACHE: dict = {}


def _pad_bufs(bs: int, max_kv: int, h: int, d: int, dtype, device):
    mk = ((max_kv + 255) // 256) * 256
    key = (bs, mk, h, d, str(dtype), str(device))
    bufs = _PAD_CACHE.get(key)
    if bufs is None:
        k = torch.zeros((bs, mk, h, d), dtype=dtype, device=device)
        v = torch.zeros_like(k)
        if len(_PAD_CACHE) > 8:
            _PAD_CACHE.clear()
        _PAD_CACHE[key] = (k, v)
        bufs = (k, v)
    return bufs



def _pad_q(
    q: torch.Tensor, cu_q: torch.Tensor, max_q: int
) -> torch.Tensor:
    """packed q [total, h, d] → padded [bs, max_q, h, d]（短序列补零）。"""
    bs = cu_q.shape[0] - 1
    out = torch.zeros((bs, max_q, q.shape[1], q.shape[2]),
                      dtype=q.dtype, device=q.device)
    off = 0
    for i in range(bs):
        n = int(cu_q[i + 1]) - int(cu_q[i])
        if n > 0:
            out[i, :n] = q[off : off + n]
        off += n
    return out


def _unpack_q(o_pad: torch.Tensor, cu_q: torch.Tensor) -> torch.Tensor:
    """padded 输出 → packed [total, h, d]。"""
    parts = []
    for i in range(o_pad.shape[0]):
        n = int(cu_q[i + 1]) - int(cu_q[i])
        if n > 0:
            parts.append(o_pad[i, :n])
    return torch.cat(parts, dim=0)


def _torch_attention(
    q: torch.Tensor,
    k_fresh: torch.Tensor,
    v_fresh: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    scale: Optional[torch.Tensor],
    is_prefill: bool,
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    max_seqlen_q: int,
    max_seqlen_k: int,
    block_tables: Optional[torch.Tensor],
    cache_seqlens: Optional[torch.Tensor],
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """SDPA 通路：gather 成 padded batch，attn_mask 屏蔽 padding 与 causal 越界。"""
    device = q.device
    block_size = k_cache.shape[1]

    if is_prefill:
        q_lens = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).to(torch.int32)
        kv_lens = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).to(torch.int32)
        max_q = max_seqlen_q
        max_kv = max_seqlen_k
    else:
        q_lens = torch.ones(cache_seqlens.shape[0], dtype=torch.int32, device=device)
        kv_lens = cache_seqlens.to(torch.int32)
        max_q = 1
        max_kv = max_seqlen_k

    # gather 页池（fp8 时反量化）；裸 prefill（无页表）直接用新算的 k/v
    if block_tables is not None:
        bufs = _pad_bufs(kv_lens.shape[0], max_kv, num_kv_heads, head_dim,
                         q.dtype, device)
        k_full, v_full = gather_kv_padded(
            k_cache, v_cache, block_tables, kv_lens, max_kv,
            q.dtype, block_size, scale, out=bufs,
        )
        k_pad, v_pad = k_full[:, :max_kv], v_full[:, :max_kv]
    else:
        # k/v packed varlen → padded
        bs = q_lens.shape[0]
        k_pad = torch.zeros((bs, max_kv, num_kv_heads, head_dim),
                            dtype=q.dtype, device=device)
        v_pad = torch.zeros_like(k_pad)
        off = 0
        for i in range(bs):
            n = int(kv_lens[i])
            if n > 0:
                k_pad[i, :n] = k_fresh[off : off + n]
                v_pad[i, :n] = v_fresh[off : off + n]
            off += n

    q_pad = (
        q.view(-1, num_qo_heads, head_dim).unsqueeze(1)  # [bs, 1, h, d]
        if not is_prefill
        else _pad_q(q.view(-1, num_qo_heads, head_dim), cu_seqlens_q, max_q)
    )

    rows = torch.arange(max_q, device=device)[None, :, None]
    cols = torch.arange(max_kv, device=device)[None, None, :]
    # PyTorch bool mask：True=参与注意力；屏蔽 causal 越界与 padding。
    keep = (cols <= (kv_lens[:, None, None] - q_lens[:, None, None] + rows)) & (
        (rows < q_lens[:, None, None]) & (cols < kv_lens[:, None, None])
    )
    mask = keep[:, None]  # [bs, 1, max_q, max_kv]

    o = F.scaled_dot_product_attention(
        q_pad.transpose(1, 2),          # [bs, h, max_q, d]
        k_pad.transpose(1, 2),          # [bs, hkv, max_kv, d]
        v_pad.transpose(1, 2),
        attn_mask=mask,
        enable_gqa=(num_qo_heads != num_kv_heads),
    )
    o = o.transpose(1, 2)               # [bs, max_q, h, d]
    if is_prefill:
        return _unpack_q(o, cu_seqlens_q).reshape(-1, num_qo_heads * head_dim)
    return o[:, 0].reshape(-1, num_qo_heads * head_dim)


# ------------------------------------------------------------------ #
# 统一入口（custom op）
# ------------------------------------------------------------------ #


@torch.library.custom_op("vllm::attention", mutates_args=())
def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    slot_mapping: torch.Tensor,
    is_prefill: bool,
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    max_seqlen_q: int,
    max_seqlen_k: int,
    block_tables: Optional[torch.Tensor],
    cache_seqlens: Optional[torch.Tensor],
    layer_id: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    meta = _ATTN_META
    assert meta is not None, "attention_context() not active"
    pool = meta.kv_cache
    k_cache = pool[0, layer_id]
    v_cache = pool[1, layer_id]
    is_fp8 = pool.dtype == torch.float8_e4m3fn
    scale = meta.kv_scale[layer_id] if is_fp8 else None
    store_kvcache(
        k,
        v,
        k_cache.reshape(-1, num_kv_heads, head_dim),
        v_cache.reshape(-1, num_kv_heads, head_dim),
        slot_mapping,
        scale,
    )
    q = q.contiguous().view(-1, num_qo_heads, head_dim)

    if _BACKEND == "torch":
        return _torch_attention(
            q, k, v, k_cache, v_cache, scale,
            is_prefill, cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k, block_tables, cache_seqlens,
            num_qo_heads, num_kv_heads, head_dim,
        )

    # fa2 后端（可选，支持 CUDA Graph）
    varlen_func, kvcache_func = _flash_attn()
    if is_fp8:
        cu_k = cu_seqlens_k if is_prefill else _cumsum(cache_seqlens)
        kg, vg = gather_fp8_kv(
            k_cache.reshape(-1, num_kv_heads, head_dim),
            v_cache.reshape(-1, num_kv_heads, head_dim),
            scale,
            block_tables,
            cu_k,
            q.dtype,
            k_cache.shape[1],
        )
        cu_q = cu_seqlens_q if is_prefill else _arange_bs(cache_seqlens)
        o = varlen_func(
            q, kg, vg,
            cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            max_seqlen_q=max_seqlen_q if is_prefill else 1,
            max_seqlen_k=max_seqlen_k,
            causal=True,
        )
        return o.view(-1, num_qo_heads * head_dim)
    if is_prefill:
        if block_tables is not None:
            k2, v2 = k_cache, v_cache
        else:
            k2 = k.contiguous().view(-1, num_kv_heads, head_dim)
            v2 = v.contiguous().view(-1, num_kv_heads, head_dim)
        o = varlen_func(
            q, k2, v2,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            causal=True,
            block_table=block_tables,
        )
    else:
        o = kvcache_func(
            q.unsqueeze(1),
            k_cache,
            v_cache,
            cache_seqlens=cache_seqlens,
            block_table=block_tables,
            causal=True,
        )
    return o.view(-1, num_qo_heads * head_dim)


def _cumsum(seqlens: torch.Tensor) -> torch.Tensor:
    device = seqlens.device
    out = torch.zeros(seqlens.shape[0] + 1, dtype=torch.int32, device=device)
    torch.cumsum(seqlens, dim=0, out=out[1:])
    return out


def _arange_bs(seqlens: torch.Tensor) -> torch.Tensor:
    return torch.arange(
        seqlens.shape[0] + 1, dtype=torch.int32, device=seqlens.device
    )


@attention.register_fake
def _attention_fake(
    q, k, v, slot_mapping, is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
    max_seqlen_k, block_tables, cache_seqlens, layer_id, num_qo_heads,
    num_kv_heads, head_dim,
):
    return torch.empty(
        (q.shape[0], num_qo_heads * head_dim), dtype=q.dtype, device=q.device
    )


class Attention(nn.Module):
    def __init__(
        self, num_heads: int, head_dim: int, num_kv_heads: int, layer_id: int = 0
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.layer_id = layer_id

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn: AttnInputs
    ) -> torch.Tensor:
        return torch.ops.vllm.attention(
            q,
            k,
            v,
            attn.slot_mapping,
            attn.is_prefill,
            attn.cu_seqlens_q,
            attn.cu_seqlens_k,
            attn.max_seqlen_q,
            attn.max_seqlen_k,
            attn.block_tables,
            attn.cache_seqlens,
            self.layer_id,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
        )
