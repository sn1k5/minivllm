from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import NamedTuple, Optional

import torch
import torch.nn as nn

from ..kv_ops import gather_fp8_kv, store_kvcache


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


@contextmanager
def attention_context(meta: AttnMetadata):
    global _ATTN_META
    prev = _ATTN_META
    _ATTN_META = meta
    try:
        yield
    finally:
        _ATTN_META = prev


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
    varlen_func, kvcache_func = _flash_attn()
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
    if is_fp8:
        # C2：FP8 页池走"反量化 gather + varlen"，prefill/decode 统一。
        return _fp8_attention(
            q, k_cache, v_cache, scale,
            is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
            block_tables, cache_seqlens,
            num_kv_heads, head_dim,
        ).view(-1, num_qo_heads * head_dim)
    if is_prefill:
        if block_tables is not None:
            k, v = k_cache, v_cache
        else:
            k = k.contiguous().view(-1, num_kv_heads, head_dim)
            v = v.contiguous().view(-1, num_kv_heads, head_dim)
        o = varlen_func(
            q,
            k,
            v,
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


def _fp8_attention(
    q, k_cache, v_cache, scale,
    is_prefill: bool, cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q: int, max_seqlen_k: int,
    block_tables, cache_seqlens,
    num_kv_heads: int, head_dim: int,
) -> torch.Tensor:
    """FP8 路径：gather 反量化成连续 KV，再走 varlen attention。"""
    device = q.device
    block_size = k_cache.shape[1]
    if is_prefill:
        cu_q = cu_seqlens_q
        cu_k = cu_seqlens_k
        max_q = max_seqlen_q
        max_k = max_seqlen_k
    else:
        bs = cache_seqlens.shape[0]
        cu_q = torch.arange(bs + 1, dtype=torch.int32, device=device)
        cu_k = torch.zeros(bs + 1, dtype=torch.int32, device=device)
        torch.cumsum(cache_seqlens, dim=0, out=cu_k)
        max_q = 1
        max_k = max_seqlen_k  # CG 场景用 context_len 上界
    k, v = gather_fp8_kv(
        k_cache.reshape(-1, num_kv_heads, head_dim),
        v_cache.reshape(-1, num_kv_heads, head_dim),
        scale,
        block_tables,
        cu_k,
        q.dtype,
        block_size,
    )
    varlen_func, _ = _flash_attn()
    return varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max_q,
        max_seqlen_k=max_k,
        causal=True,
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
