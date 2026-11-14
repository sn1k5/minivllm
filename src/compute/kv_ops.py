from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        return
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < D
    key_offsets = idx * key_stride + offsets
    value_offsets = idx * value_stride + offsets
    key = tl.load(key_ptr + key_offsets, mask=mask, other=0.0)
    value = tl.load(value_ptr + value_offsets, mask=mask, other=0.0)
    cache_offsets = slot * D + offsets
    tl.store(k_cache_ptr + cache_offsets, key, mask=mask)
    tl.store(v_cache_ptr + cache_offsets, value, mask=mask)


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    scale: torch.Tensor | None = None,
) -> None:
    """KV 写入页池。scale 非 None 时走 FP8 量化路径（per-token scale）。"""
    if not key.is_contiguous():
        key = key.contiguous()
    if not value.is_contiguous():
        value = value.contiguous()

    n, num_heads, head_dim = key.shape
    d = num_heads * head_dim
    assert slot_mapping.numel() == n

    if scale is not None:
        block = triton.next_power_of_2(d)
        _store_kvcache_fp8_kernel[(n,)](
            key, key.stride(0), value, value.stride(0),
            k_cache, v_cache, slot_mapping, scale,
            d, block,
        )
        return

    block = triton.next_power_of_2(d)
    _store_kvcache_kernel[(n,)](
        key,
        key.stride(0),
        value,
        value.stride(0),
        k_cache,
        v_cache,
        slot_mapping,
        d,
        block,
    )


@triton.jit
def _store_kvcache_fp8_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    scale_ptr,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """FP8 写入：以 key 的 amax 定 per-token scale，K/V 共用同一 scale。"""
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        return
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < D
    key = tl.load(key_ptr + idx * key_stride + offsets, mask=mask, other=0.0)
    amax = tl.max(tl.abs(key), axis=0)
    s = tl.maximum(amax / 448.0, 1e-8)
    k8 = (key / s).to(k_cache_ptr.dtype.element_ty)
    value = tl.load(value_ptr + idx * value_stride + offsets, mask=mask, other=0.0)
    v8 = (value / s).to(v_cache_ptr.dtype.element_ty)
    cache_offsets = slot * D + offsets
    tl.store(k_cache_ptr + cache_offsets, k8, mask=mask)
    tl.store(v_cache_ptr + cache_offsets, v8, mask=mask)
    tl.store(scale_ptr + slot, s)


@triton.jit
def _gather_fp8_kv_kernel(
    k_cache_ptr,
    v_cache_ptr,
    scale_ptr,
    bt_ptr,
    bt_stride,
    cu_ptr,
    k_out_ptr,
    v_out_ptr,
    D: tl.constexpr,
    BS: tl.constexpr,
):
    """按页表把 FP8 页拉直成连续 KV，并按 per-token scale 反量化。"""
    sid = tl.program_id(0)
    blk = tl.program_id(1)
    start = tl.load(cu_ptr + sid)
    end = tl.load(cu_ptr + sid + 1)
    cache_len = end - start
    if blk * BS >= cache_len:
        return
    bid = tl.load(bt_ptr + sid * bt_stride + blk)
    if bid < 0:
        return
    n_tok = tl.minimum(BS, cache_len - blk * BS)
    offs_t = tl.arange(0, BS)
    offs_d = tl.arange(0, D)
    m = offs_t[:, None] < n_tok
    slot0 = bid * BS
    base = slot0 * D + offs_t[:, None] * D + offs_d[None, :]
    k8 = tl.load(k_cache_ptr + base, mask=m, other=0.0)
    v8 = tl.load(v_cache_ptr + base, mask=m, other=0.0)
    s = tl.load(scale_ptr + slot0 + offs_t, mask=offs_t < n_tok, other=1.0)
    out_base = (start + blk * BS + offs_t)[:, None] * D + offs_d[None, :]
    sf = s.to(tl.float32)[:, None]
    tl.store(
        k_out_ptr + out_base,
        (k8.to(tl.float32) * sf).to(k_out_ptr.dtype.element_ty),
        mask=m,
    )
    tl.store(
        v_out_ptr + out_base,
        (v8.to(tl.float32) * sf).to(v_out_ptr.dtype.element_ty),
        mask=m,
    )


def gather_fp8_kv(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    scale: torch.Tensor,
    block_tables: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    dtype: torch.dtype,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP8 页池 → 连续 KV（反量化）。block_tables [bs, mb]，cu_k [bs+1]。"""
    n_seq = block_tables.shape[0]
    d = k_cache.shape[1] * k_cache.shape[2]
    total = int(cu_seqlens_k[-1].item())
    k_out = torch.empty((total, k_cache.shape[1], k_cache.shape[2]),
                        dtype=dtype, device=k_cache.device)
    v_out = torch.empty_like(k_out)
    grid = (n_seq, block_tables.shape[1])
    _gather_fp8_kv_kernel[grid](
        k_cache, v_cache, scale, block_tables, block_tables.stride(0),
        cu_seqlens_k, k_out, v_out,
        D=d, BS=block_size,
    )
    return k_out, v_out
