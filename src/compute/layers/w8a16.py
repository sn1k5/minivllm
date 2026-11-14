from __future__ import annotations

import torch
import triton
import triton.language as tl

# ==============================================================================
# W8A16 权重量化：int8 权重（per-output-channel scale）× bf16/fp16 激活。
#
# 量化在权重加载时在线完成（quantize-on-load），无需离线校准：
#   每个输出通道算 absmax → scale = amax / 127 → q = round(w / scale)。
# 前向走下面的 Triton dequant GEMM：int8 tile 转 fp 后乘 scale 再 tl.dot。
# 用 module 级开关而非改构造函数签名，避免把量化参数穿透所有模型代码。
# ==============================================================================

_ENABLED = False


def set_w8a16_enabled(on: bool) -> None:
    global _ENABLED
    _ENABLED = on


def w8a16_enabled() -> bool:
    return _ENABLED


def quantize_channelwise(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """按输出通道量化：返回 (int8 权重, fp16 scale)，scale 与第 0 维对齐。"""
    amax = w.abs().amax(dim=1)
    scale = (amax / 127.0).clamp(min=1e-8).to(torch.float16)
    q = (
        w.to(torch.float32)
        / scale.to(torch.float32)[:, None]
    ).round().clamp(-127, 127).to(torch.int8)
    return q, scale


@triton.jit
def _w8a16_gemm_kernel(
    x_ptr, w_ptr, s_ptr, out_ptr,
    T, N, K,
    sx0, sx1, so0,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_t = pid_t * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        offs_k = k0 + tl.arange(0, BK)
        x = tl.load(
            x_ptr + offs_t[:, None] * sx0 + offs_k[None, :] * sx1,
            mask=(offs_t[:, None] < T) & (offs_k[None, :] < K),
            other=0.0,
        )
        wq = tl.load(
            w_ptr + offs_n[:, None] * K + offs_k[None, :],
            mask=(offs_n[:, None] < N) & (offs_k[None, :] < K),
            other=0,
        )
        w = wq.to(x.dtype)
        s = tl.load(s_ptr + offs_n, mask=offs_n < N, other=0.0)
        w = w * s[:, None].to(x.dtype)
        acc += tl.dot(x, tl.trans(w))
    out = acc.to(out_ptr.dtype.element_ty)
    tl.store(
        out_ptr + offs_t[:, None] * so0 + offs_n[None, :],
        out,
        mask=(offs_t[:, None] < T) & (offs_n[None, :] < N),
    )


_BM, _BN, _BK = 64, 64, 64


def w8a16_linear(
    x: torch.Tensor,
    qweight: torch.Tensor,
    qscale: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """dequant GEMM：out = x @ (qweight * scale)^T (+ bias)。"""
    orig_shape = x.shape
    x2 = x.reshape(-1, orig_shape[-1]).contiguous()
    T, K = x2.shape
    N = qweight.shape[0]
    out = torch.empty((T, N), dtype=x.dtype, device=x.device)
    if T == 0:
        return out.reshape(*orig_shape[:-1], N)
    grid = (triton.cdiv(T, _BM), triton.cdiv(N, _BN))
    _w8a16_gemm_kernel[grid](
        x2, qweight, qscale, out,
        T, N, K,
        x2.stride(0), x2.stride(1), out.stride(0),
        BM=_BM, BN=_BN, BK=_BK,
        num_warps=4,
    )
    if bias is not None:
        out = out + bias
    return out.reshape(*orig_shape[:-1], N)
