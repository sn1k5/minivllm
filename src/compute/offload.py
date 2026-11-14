from __future__ import annotations

from collections import deque

import torch

from .kv_pool import KVPool

# ==============================================================================
# 主机 KV 池后端：pinned memory 槽位 + D2H/H2D 拷贝。
#
# 每个"槽位"是独立分配的一小块 pinned 张量（与 GPU 单页同形），
# 按需分配，不预占整机内存。swap 区域只是若干槽位的列表，不要求连续。
# 拷贝走独立 CUDA stream：H2D 异步发射，event 上报就绪。
# ==============================================================================


class HostKVBackend:
    def __init__(self, kv_pool: KVPool, *, host_max_blocks: int) -> None:
        kv = kv_pool.kv_cache  # [2, layers, num_blocks, block_size, nkv, hd]
        _, layers, _, block_size, nkv, hd = kv.shape
        self.kv_cache = kv
        self.dtype = kv.dtype
        self.host_max = host_max_blocks
        self.slots: list[torch.Tensor | None] = [None] * host_max_blocks
        self.free_slots: deque[int] = deque(range(host_max_blocks))
        self.stream = torch.cuda.Stream()
        self.shape = (2, layers, block_size, nkv, hd)

    # ------------------------------------------------------------------ #
    # 槽位
    # ------------------------------------------------------------------ #

    def alloc_slot(self) -> int:
        if not self.free_slots:
            return -1
        return self.free_slots.popleft()

    def free_slot(self, slot: int) -> None:
        self.slots[slot] = None
        self.free_slots.append(slot)

    def alloc_region(self, n: int) -> list[int]:
        if len(self.free_slots) < n:
            return []
        return [self.free_slots.popleft() for _ in range(n)]

    def free_region(self, slots: list[int]) -> None:
        for s in slots:
            self.free_slot(s)

    def _slot_tensor(self, slot: int) -> torch.Tensor:
        t = self.slots[slot]
        if t is None:
            t = torch.empty(self.shape, dtype=self.dtype, pin_memory=True)
            self.slots[slot] = t
        return t

    # ------------------------------------------------------------------ #
    # 拷贝
    # ------------------------------------------------------------------ #

    def copy_out(self, gpu_bid: int, slot: int) -> None:
        """整页 D2H（同步，调度期调用，此时计算流空闲）。"""
        self._slot_tensor(slot).copy_(self.kv_cache[:, :, gpu_bid])

    def copy_out_region(self, gpu_bids: list[int], slots: list[int]) -> None:
        for bid, slot in zip(gpu_bids, slots):
            self.copy_out(bid, slot)

    def copy_in_async(self, host_slots: list[int], gpu_bids: list[int]) -> torch.cuda.Event:
        """H2D 异步发射（pinned 源），返回事件供轮询。"""
        cur = torch.cuda.current_stream()
        self.stream.wait_stream(cur)
        with torch.cuda.stream(self.stream):
            for h, g in zip(host_slots, gpu_bids):
                self.kv_cache[:, :, g].copy_(self._slot_tensor(h), non_blocking=True)
        event = torch.cuda.Event()
        event.record(self.stream)
        return event

    @staticmethod
    def event_ready(event: torch.cuda.Event) -> bool:
        return event.query()

    @staticmethod
    def event_synchronize(event: torch.cuda.Event) -> None:
        event.synchronize()
