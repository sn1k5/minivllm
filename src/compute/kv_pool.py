from __future__ import annotations

import torch
from transformers import PretrainedConfig

from ..distributed import all_gather, get_pp_group, get_pp_indices, get_tp_group


def get_kv_dims(hf: PretrainedConfig) -> tuple[int, int, int]:
    tp = get_tp_group()
    pp = get_pp_group()
    num_kv_heads = max(1, hf.num_key_value_heads // tp.size)
    start, end = get_pp_indices(hf.num_hidden_layers, pp.group_rank, pp.size)
    head_dim = getattr(hf, "head_dim", hf.hidden_size // hf.num_attention_heads)
    return end - start, num_kv_heads, head_dim


class KVPool:
    def __init__(
        self,
        hf_config: PretrainedConfig,
        *,
        num_blocks: int,
        block_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be > 0, got {num_blocks}")
        self.hf_config = hf_config
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.device = device
        self.dtype = dtype
        self.kv_cache = self._alloc_pool(
            hf_config, num_blocks, block_size, device, dtype
        )
        # C2 KV FP8：per-(layer, slot) 的量化 scale；普通精度下为 None。
        layers, _, _ = get_kv_dims(hf_config)
        self.kv_scale: torch.Tensor | None = None
        if dtype == torch.float8_e4m3fn:
            self.kv_scale = torch.ones(
                layers, num_blocks * block_size, dtype=torch.float32, device=device
            )

    @staticmethod
    def _alloc_pool(
        hf: PretrainedConfig,
        num_blocks: int,
        block_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        layers, nkv, hd = get_kv_dims(hf)
        block_bytes = 2 * layers * block_size * nkv * hd * dtype.itemsize
        kv = torch.empty(
            2, layers, num_blocks, block_size, nkv, hd, dtype=dtype, device=device
        )
        print(
            f"KV pool: {num_blocks} blocks × {block_size} tok "
            f"({num_blocks * block_bytes / (1024**3):.2f} GiB)"
        )
        return kv

    @property
    def num_blocks_total(self) -> int:
        return self.num_blocks

# 先测出真实可用显存 → 扣掉 CUDA Graph 预留 → 按单块字节数换算成 block 数 → 多卡取最小值对齐。
# 自动计算 GPU 上能放下多少个 KV Cache block
    @classmethod
    def auto_num_blocks(
        cls,
        hf: PretrainedConfig,
        *,
        block_size: int,
        dtype: torch.dtype,
        device: torch.device,
        gpu_memory_utilization: float,
        reserve_bytes: int = 0,
    ) -> int:
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        layers, nkv, hd = get_kv_dims(hf)
        block_bytes = 2 * layers * block_size * nkv * hd * dtype.itemsize
        free, total = torch.cuda.mem_get_info(device)
        available = int(total * gpu_memory_utilization) - (total - free)
        available -= reserve_bytes
        num_blocks = max(4, available // block_bytes)
        return int(
            all_gather(torch.tensor([num_blocks], device=device), dim=0).min().item()
        )
