from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

# ==============================================================================
# KV offload 账本：GPU 页被前缀索引淘汰时，把 KV 拷到主机内存留档。
#
# 职责（本类只记账，不碰 torch）：
#   - 主机槽位的分配与 LRU 回收。
#   - 链哈希 → 主机槽位 的二级索引，命中已换出前缀时定位。
#   - 拉回任务（异步 H2D）的登记与就绪轮询。
#   - 整条序列的 swap_out / swap_in（抢占用，见 A1 文档）。
#
# 真正的拷贝由 compute/offload.py 的 HostKVBackend 完成，
# 通过 attach_backend 注入（duck-typed，无 import 依赖）。
# 仅支持单机 tp=pp=1（InprocExecutor），多进程执行器不启用。
# ==============================================================================


@dataclass
class RestoreJob:
    """一次异步拉回：主机槽位 → 新分配的 GPU 页。就绪前序列不能进前向。"""

    seq_id: int
    n_blocks: int
    host_slots: list[int]
    gpu_bids: list[int]
    kind: str = "prefix"  # prefix=前缀命中拉回 / swap=抢占换入
    event: Any = None
    started_at: float = field(default_factory=time.perf_counter)


class KVOffloader:
    def __init__(self, *, host_max_blocks: int) -> None:
        self.host_max_blocks = host_max_blocks
        self.backend = None  # HostKVBackend，由 Executor 注入
        # 链哈希 → 主机槽位；GPU 侧淘汰时留档，前缀命中时查这里。
        self.hash_to_host: dict[int, int] = {}
        # 主机槽位 LRU：值是该槽对应的链哈希（swap 区域不进 LRU）。
        self.host_lru: OrderedDict[int, int] = OrderedDict()
        # swap 区域：seq_id → 主机槽位列表。
        self.swap_slots: dict[int, list[int]] = {}
        self.restoring: dict[int, RestoreJob] = {}
        # 统计
        self.n_offloaded = 0
        self.n_host_hits = 0
        self.n_restored = 0
        self.n_swap_out = 0
        self.n_swap_in = 0

    # ------------------------------------------------------------------ #
    # 后端注入
    # ------------------------------------------------------------------ #

    def attach_backend(self, backend) -> None:
        self.backend = backend

    @property
    def enabled(self) -> bool:
        return self.backend is not None

    @property
    def num_host_used(self) -> int:
        n = len(self.hash_to_host)
        n += sum(len(v) for v in self.swap_slots.values())
        n += sum(j.n_blocks for j in self.restoring.values())
        return n

    # ------------------------------------------------------------------ #
    # 主机槽位管理
    # ------------------------------------------------------------------ #

    def _alloc_host_slot(self) -> int:
        slot = self.backend.alloc_slot()
        if slot >= 0:
            return slot
        return self._evict_host_lru()

    def _evict_host_lru(self) -> int:
        """挤掉最久未用的留档页，返回腾出的槽位；挤不动返回 -1。"""
        while self.host_lru:
            slot, h = self.host_lru.popitem(last=False)
            if self.hash_to_host.get(h) == slot:
                del self.hash_to_host[h]
                self.backend.free_slot(slot)
                return slot
        return -1

    def _alloc_host_region(self, n: int) -> list[int]:
        slots = self.backend.alloc_region(n)
        if slots:
            return slots
        while self.host_lru and self._evict_host_lru() >= 0:
            slots = self.backend.alloc_region(n)
            if slots:
                return slots
        return []

    # ------------------------------------------------------------------ #
    # GPU 淘汰 → 主机留档
    # ------------------------------------------------------------------ #

    def preserve(self, bid: int, block_hash: int) -> bool:
        """GPU 页被淘汰前调用：拷到主机并登记链哈希。失败返回 False。"""
        if not self.enabled or block_hash == -1:
            return False
        if block_hash in self.hash_to_host:
            return True
        slot = self._alloc_host_slot()
        if slot < 0:
            return False
        self.backend.copy_out(bid, slot)
        self.hash_to_host[block_hash] = slot
        self.host_lru[slot] = block_hash
        self.n_offloaded += 1
        return True

    # ------------------------------------------------------------------ #
    # 前缀命中 → 异步拉回
    # ------------------------------------------------------------------ #

    def lookup_host(self, block_hash: int) -> int:
        """查留档：返回主机槽位，未命中返回 -1。"""
        slot = self.hash_to_host.get(block_hash, -1)
        if slot >= 0:
            self.host_lru.move_to_end(slot)
            self.n_host_hits += 1
        return slot

    def start_restore(
        self, seq_id: int, host_slots: list[int], gpu_bids: list[int], *, kind: str
    ) -> RestoreJob:
        """发起异步 H2D，登记任务。gpu_bids 已在 BlockManager 分配好。"""
        job = RestoreJob(seq_id=seq_id, n_blocks=len(host_slots),
                         host_slots=host_slots, gpu_bids=gpu_bids, kind=kind)
        job.event = self.backend.copy_in_async(host_slots, gpu_bids)
        self.restoring[seq_id] = job
        return job

    def poll_restore(self, seq_id: int) -> bool:
        """拉回是否就绪。就绪后移除任务并计数。"""
        job = self.restoring.get(seq_id)
        if job is None:
            return True
        if self.backend.event_ready(job.event):
            del self.restoring[seq_id]
            self.n_restored += 1
            return True
        return False

    def abort_restore(self, seq_id: int) -> list[int]:
        """放弃拉回：等拷贝结束后释放 GPU 页，返回这些页号。"""
        job = self.restoring.pop(seq_id, None)
        if job is None:
            return []
        self.backend.event_synchronize(job.event)
        return job.gpu_bids

    # ------------------------------------------------------------------ #
    # 整条序列的 swap（A1 抢占）
    # ------------------------------------------------------------------ #

    def swap_out(self, seq_id: int, gpu_bids: list[int]) -> bool:
        """整条序列换出：全部页拷到主机。成功返回 True。"""
        if not self.enabled:
            return False
        slots = self._alloc_host_region(len(gpu_bids))
        if not slots:
            return False
        self.backend.copy_out_region(gpu_bids, slots)
        self.swap_slots[seq_id] = slots
        self.n_swap_out += 1
        return True

    def swap_in(self, seq_id: int, gpu_bids: list[int]) -> RestoreJob | None:
        """换入：发起异步 H2D。gpu_bids 为新分配的 GPU 页。"""
        slots = self.swap_slots.pop(seq_id, None)
        if slots is None:
            return None
        job = RestoreJob(seq_id=seq_id, n_blocks=len(slots),
                         host_slots=slots, gpu_bids=gpu_bids, kind="swap")
        job.event = self.backend.copy_in_async(slots, gpu_bids)
        self.restoring[seq_id] = job
        self.n_swap_in += 1
        return job

    def drop_swap(self, seq_id: int) -> None:
        """序列被放弃时清理 swap 区域。"""
        slots = self.swap_slots.pop(seq_id, None)
        if slots:
            self.backend.free_region(slots)

    # ------------------------------------------------------------------ #
    # 观测
    # ------------------------------------------------------------------ #

    def stats(self) -> dict:
        return {
            "host_blocks_used": self.num_host_used,
            "host_blocks_max": self.host_max_blocks,
            "offloaded_blocks": self.n_offloaded,
            "host_hits": self.n_host_hits,
            "restored_blocks": self.n_restored,
            "restoring_seqs": len(self.restoring),
            "swap_out_seqs": self.n_swap_out,
            "swap_in_seqs": self.n_swap_in,
        }
