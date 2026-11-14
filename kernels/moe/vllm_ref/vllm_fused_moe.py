# 改自vllm v0.6.3 fused_moe.py, 去掉vllm依赖, 仅用于对比测试
# 原文件: vllm/model_executor/layers/fused_moe/fused_moe.py
# kernel部分保持原样, 依赖替换成torch/triton等价实现
import functools
import os
import sys
from typing import Any, Dict, Optional, Tuple

import torch
import triton
import triton.language as tl

# 优先用CUDA版align, 和vllm原版CUDA op开销对齐
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
try:
    from align_ops import moe_align as _moe_align_cuda
except Exception:
    _moe_align_cuda = None

CHUNK_SIZE = 32768  # 原envs.VLLM_FUSED_MOE_CHUNK_SIZE默认值


def _moe_align_block_size_torch(topk_ids, num_experts, block_size,
                                sorted_ids, expert_ids, num_tokens_post_pad):
    # 纯torch实现, 语义对齐vllm的CUDA op
    flat = topk_ids.view(-1).long()
    counts = torch.bincount(flat, minlength=num_experts)
    pads = (counts + block_size - 1) // block_size * block_size
    poff = torch.cumsum(pads, 0)
    cnt_start = torch.cumsum(counts, 0) - counts
    pad_start = poff - pads
    order = flat.argsort(stable=True)
    idx = torch.arange(flat.numel(), device=flat.device)
    se = flat[order]
    pos = pad_start[se] + (idx - cnt_start[se])
    sorted_ids.fill_(topk_ids.numel())
    sorted_ids[pos] = order.to(torch.int32)
    num_tokens_post_pad[0] = poff[-1].to(torch.int32)
    nblk = (pads // block_size).long()
    nb = int(nblk.sum())
    expert_ids.fill_(0)
    expert_ids[:nb] = torch.repeat_interleave(
        torch.arange(num_experts, device=flat.device), nblk).to(torch.int32)


def moe_align_block_size(topk_ids: torch.Tensor, block_size: int,
                         num_experts: int):
    if _moe_align_cuda is not None:
        return _moe_align_cuda(topk_ids, block_size, num_experts)
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=torch.int32,
                             device=topk_ids.device)
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    expert_ids = torch.empty((max_num_m_blocks,), dtype=torch.int32,
                             device=topk_ids.device)
    num_tokens_post_pad = torch.empty((1,), dtype=torch.int32,
                                      device=topk_ids.device)
    _moe_align_block_size_torch(topk_ids, num_experts, block_size, sorted_ids,
                                expert_ids, num_tokens_post_pad)
    return sorted_ids, expert_ids, num_tokens_post_pad


# 替代vllm的CUDA silu_and_mul op, 语义一致
@triton.jit
def _silu_mul_kernel(x_ptr, out_ptr, N, total, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    row = offs // N
    col = offs % N
    g = tl.load(x_ptr + row * 2 * N + col, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(x_ptr + row * 2 * N + N + col, mask=mask, other=0.0).to(tl.float32)
    s = g * tl.sigmoid(g)
    tl.store(out_ptr + offs, (s * u).to(out_ptr.dtype.element_ty), mask=mask)


def silu_and_mul(out, x):
    # x [..., 2N] -> out [..., N]
    N = x.shape[-1] // 2
    total = x.numel() // 2
    BLOCK = 1024
    _silu_mul_kernel[(triton.cdiv(total, BLOCK),)](x, out, N, total, BLOCK)


@triton.jit
def fused_moe_kernel(
        a_ptr, b_ptr, c_ptr, a_scale_ptr, b_scale_ptr,
        topk_weights_ptr, sorted_token_ids_ptr, expert_ids_ptr,
        num_tokens_post_padded_ptr,
        N, K, EM, num_valid_tokens,
        stride_am, stride_ak,
        stride_be, stride_bk, stride_bn,
        stride_cm, stride_cn,
        stride_bse, stride_bsn,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
        MUL_ROUTED_WEIGHT: tl.constexpr, top_k: tl.constexpr,
        compute_type: tl.constexpr, use_fp8_w8a8: tl.constexpr,
        use_int8_w8a16: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am +
                      offs_k[None, :] * stride_ak)
    off_experts = tl.load(expert_ids_ptr + pid_m)
    b_ptrs = b_ptr + off_experts * stride_be + (offs_k[:, None] * stride_bk +
                                                offs_bn[None, :] * stride_bn)
    if use_int8_w8a16:
        b_scale_ptrs = b_scale_ptr + off_experts * stride_bse + offs_bn[
            None, :] * stride_bsn
        b_scale = tl.load(b_scale_ptrs)
    if use_fp8_w8a8:
        a_scale = tl.load(a_scale_ptr)
        b_scale = tl.load(b_scale_ptr + off_experts)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs,
                    mask=token_mask[:, None] &
                    (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                    other=0.0)
        b = tl.load(b_ptrs,
                    mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
                    other=0.0)
        if use_int8_w8a16:
            accumulator = tl.dot(a, b.to(compute_type), acc=accumulator)
        elif use_fp8_w8a8:
            accumulator = tl.dot(a, b, acc=accumulator)
        else:
            accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token,
                             mask=token_mask,
                             other=0)
        accumulator = accumulator * moe_weight[:, None]
    if use_int8_w8a16:
        accumulator = (accumulator * b_scale).to(compute_type)
    elif use_fp8_w8a8:
        accumulator = (accumulator * a_scale * b_scale).to(compute_type)
    else:
        accumulator = accumulator.to(compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[
        None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def invoke_fused_moe_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
                            A_scale: Optional[torch.Tensor],
                            B_scale: Optional[torch.Tensor],
                            topk_weights: torch.Tensor, topk_ids: torch.Tensor,
                            sorted_token_ids: torch.Tensor,
                            expert_ids: torch.Tensor,
                            num_tokens_post_padded: torch.Tensor,
                            mul_routed_weight: bool, top_k: int,
                            config: Dict[str, Any], compute_type: tl.dtype,
                            use_fp8_w8a8: bool, use_int8_w8a16: bool) -> None:
    assert topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1
    if use_fp8_w8a8:
        A, A_scale = A.to(torch.float8_e4m3fn), A_scale
        assert B_scale is not None
    elif use_int8_w8a16:
        assert B_scale is not None
    else:
        assert A_scale is None
        assert B_scale is None
    grid = lambda META: (triton.cdiv(sorted_token_ids.shape[0], META[
        'BLOCK_SIZE_M']) * triton.cdiv(B.shape[1], META['BLOCK_SIZE_N']), )
    fused_moe_kernel[grid](
        A, B, C, A_scale, B_scale,
        topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded,
        B.shape[1], B.shape[2],
        sorted_token_ids.shape[0], topk_ids.numel(),
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(2), B.stride(1),
        C.stride(1), C.stride(2),
        B_scale.stride(0) if B_scale is not None and use_int8_w8a16 else 0,
        B_scale.stride(1) if B_scale is not None and use_int8_w8a16 else 0,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        **config,
    )


def get_default_config(M: int, E: int, N: int, K: int, topk: int,
                       dtype: Optional[str], is_marlin: bool) -> Dict[str, int]:
    config = {
        'BLOCK_SIZE_M': 64,
        'BLOCK_SIZE_N': 64,
        'BLOCK_SIZE_K': 32,
        'GROUP_SIZE_M': 8
    }
    if M <= E or (is_marlin and M <= 32):
        config = {
            'BLOCK_SIZE_M': 16,
            'BLOCK_SIZE_N': 32,
            'BLOCK_SIZE_K': 64,
            'GROUP_SIZE_M': 1
        }
    return config


def try_get_optimal_moe_config(
    w1_shape: Tuple[int, ...],
    w2_shape: Tuple[int, ...],
    top_k: int,
    dtype: Optional[str],
    M: int,
    override_config: Optional[Dict[str, Any]] = None,
    is_marlin: bool = False,
):
    if override_config:
        config = override_config
    else:
        E, _, N = w2_shape
        config = get_default_config(M, E, N, w1_shape[2], top_k, dtype,
                                    is_marlin)
    return config


def fused_experts(hidden_states: torch.Tensor,
                  w1: torch.Tensor,
                  w2: torch.Tensor,
                  topk_weights: torch.Tensor,
                  topk_ids: torch.Tensor,
                  inplace: bool = False,
                  override_config: Optional[Dict[str, Any]] = None):
    assert hidden_states.shape[1] == w1.shape[2], "Hidden size mismatch"
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16]

    num_tokens, _ = hidden_states.shape
    E, N, _ = w1.shape
    M = min(num_tokens, CHUNK_SIZE)
    config_dtype = None

    get_config_func = functools.partial(
        try_get_optimal_moe_config,
        w1.shape,
        w2.shape,
        topk_ids.shape[1],
        config_dtype,
        override_config=override_config,
    )
    config = get_config_func(M)

    intermediate_cache1 = torch.empty((M, topk_ids.shape[1], N),
                                      device=hidden_states.device,
                                      dtype=hidden_states.dtype)
    intermediate_cache2 = torch.empty((M * topk_ids.shape[1], N // 2),
                                      device=hidden_states.device,
                                      dtype=hidden_states.dtype)
    intermediate_cache3 = torch.empty((M, topk_ids.shape[1], w2.shape[1]),
                                      device=hidden_states.device,
                                      dtype=hidden_states.dtype)

    compute_type = (tl.bfloat16
                    if hidden_states.dtype == torch.bfloat16 else tl.float16)

    if inplace:
        out_hidden_states = hidden_states
    else:
        out_hidden_states = torch.empty_like(hidden_states)

    for chunk in range((num_tokens // CHUNK_SIZE) + 1):
        begin_chunk_idx, end_chunk_idx = (chunk * CHUNK_SIZE,
                                          min((chunk + 1) * CHUNK_SIZE,
                                              num_tokens))
        curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
        tokens_in_chunk, _ = curr_hidden_states.shape
        if tokens_in_chunk == 0:
            break
        if tokens_in_chunk < CHUNK_SIZE and chunk > 0:
            intermediate_cache1 = intermediate_cache1[:tokens_in_chunk]
            intermediate_cache2 = intermediate_cache2[:tokens_in_chunk]
            intermediate_cache3 = intermediate_cache3[:tokens_in_chunk]
            config = get_config_func(tokens_in_chunk)

        curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
        curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]

        sorted_token_ids, expert_ids, num_tokens_post_padded = (
            moe_align_block_size(curr_topk_ids, config['BLOCK_SIZE_M'], E))

        invoke_fused_moe_kernel(curr_hidden_states, w1,
                                intermediate_cache1, None, None,
                                curr_topk_weights, curr_topk_ids,
                                sorted_token_ids, expert_ids,
                                num_tokens_post_padded,
                                False, topk_ids.shape[1], config,
                                compute_type=compute_type,
                                use_fp8_w8a8=False, use_int8_w8a16=False)

        silu_and_mul(intermediate_cache2, intermediate_cache1.view(-1, N))

        invoke_fused_moe_kernel(intermediate_cache2, w2,
                                intermediate_cache3, None, None,
                                curr_topk_weights, curr_topk_ids,
                                sorted_token_ids, expert_ids,
                                num_tokens_post_padded,
                                True, 1, config,
                                compute_type=compute_type,
                                use_fp8_w8a8=False, use_int8_w8a16=False)

        torch.sum(intermediate_cache3.view(*intermediate_cache3.shape),
                  dim=1,
                  out=out_hidden_states[begin_chunk_idx:end_chunk_idx])
    return out_hidden_states


def fused_moe(hidden_states: torch.Tensor, w1: torch.Tensor,
              w2: torch.Tensor, gating_output: torch.Tensor, topk: int,
              renormalize: bool, inplace: bool = False,
              override_config: Optional[Dict[str, Any]] = None) -> torch.Tensor:
    # 路由用torch实现(原ops.topk_softmax不可用), 与测试基准一致
    prob = torch.softmax(gating_output.float(), dim=-1)
    topk_weights, topk_ids = torch.topk(prob, topk, dim=-1)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return fused_experts(hidden_states, w1, w2,
                         topk_weights.to(torch.float32),
                         topk_ids.to(torch.int32),
                         inplace=inplace, override_config=override_config)
