from __future__ import annotations

from .block_manager import BlockManager
from .offload import KVOffloader, RestoreJob
from .prefix import PrefixBackend, RadixPrefixIndex, make_prefix_index
from .prefix.hash_util import compute_block_hash
from .sequence import Sequence

# ==============================================================================
# driver 侧 KV 账本：给序列发逻辑页、查前缀命中、用完回收。
#
# Scheduler 调 can_allocate / allocate / may_append / deallocate。
# GPU 上真正的 cache 张量在 compute/kv_pool.py，本类不 import torch。
#
# offload（B1）：主机留档与 GPU 前缀索引联动——
#   淘汰的满块拷到主机留档（链哈希索引）；allocate 时 GPU 命中之后
#   继续查主机，命中则分配新 GPU 页并发起异步 H2D 拉回，
#   就绪前序列不进前向。仅单机 tp=pp=1 可用。
# ==============================================================================


class KVManager:
    def __init__(
        self,
        *,
        num_blocks: int,
        block_size: int,
        prefix_backend: PrefixBackend = "hash",
    ) -> None:
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be > 0, got {num_blocks}")
        self.block_size = block_size
        self.prefix_backend: PrefixBackend = prefix_backend
        self.num_blocks_total = num_blocks
        self.num_blocks_schedulable = num_blocks
        self.blocks = BlockManager(self.num_blocks_schedulable, block_size)
        self.index = make_prefix_index(prefix_backend)
        self.blocks.on_evict = lambda bid: self.index.invalidate(bid, self.blocks)
        if isinstance(self.index, RadixPrefixIndex):
            self.blocks.reclaim_fn = lambda n: self.index.reclaim_pages(n, self.blocks)
        self.offloader: KVOffloader | None = None
        self.last_num_cached_tokens = 0

    @property
    def num_free(self) -> int:
        return self.blocks.num_free

    def attach_offloader(self, offloader: KVOffloader) -> None:
        """接入 offload（账本 + 主机后端），同时让淘汰路径先留档。"""
        self.offloader = offloader
        self.blocks.offload = offloader

    # ------------------------------------------------------------------ #
    # 前缀命中
    # ------------------------------------------------------------------ #

    def _fit_prefix(self, seq: Sequence) -> int:
        """前缀复用预检查：计算当前序列最多可复用多少连续前缀 KV 块。"""
        need = seq.num_blocks
        if need == 0:
            return 0
        hit = self.index.match(seq, self.blocks)
        max_cached = min(len(hit), need)
        reclaimable = sum(
            1 for b in hit[:max_cached] if b in self.blocks.cached_ids
        )
        other_cached = len(self.blocks.cached_ids) - reclaimable
        new_needed = need - max_cached
        if self.blocks.num_free + other_cached >= new_needed:
            return max_cached
        return -1

    def probe_prefix_len(self, seq: Sequence) -> int:
        """只读探测：GPU 命中的满块数（路由器用，不分配）。"""
        return len(self.index.match(seq, self.blocks))

    def _match_host(self, seq: Sequence, gpu_hits: int) -> tuple[int, list[int]]:
        """GPU 命中之后继续匹配主机留档。返回 (命中块数, 主机槽位)。"""
        if self.offloader is None or not self.offloader.enabled:
            return 0, []
        if gpu_hits > 0:
            last_hash = self.blocks.blocks[seq.block_table[gpu_hits - 1]].hash
            if last_hash == -1:
                return 0, []
        else:
            last_hash = -1
        n = 0
        slots: list[int] = []
        for i in range(gpu_hits, seq.num_blocks):
            h = compute_block_hash(seq.block_token_ids(i), last_hash)
            slot = self.offloader.lookup_host(h)
            if slot < 0:
                break
            slots.append(slot)
            last_hash = h
            n += 1
        return n, slots

    # ------------------------------------------------------------------ #
    # 分配
    # ------------------------------------------------------------------ #

    def can_allocate(self, seq: Sequence) -> bool:
        return self._fit_prefix(seq) >= 0

    def allocate(self, seq: Sequence) -> RestoreJob | None:
        """分配页表；命中主机留档时返回待等待的 RestoreJob，否则返回 None。"""
        seq.block_table.clear()
        seq.num_cached_tokens = 0
        need = seq.num_blocks
        if need == 0:
            self.last_num_cached_tokens = 0
            return None

        max_cached = self._fit_prefix(seq)
        if max_cached < 0:
            raise RuntimeError("not enough KV blocks for sequence")
        hit = self.index.match(seq, self.blocks)

        for bid in hit[:max_cached]:
            self.blocks.acquire_hit(bid)
            seq.block_table.append(bid)

        # 主机留档接续：GPU 命中断点之后查链哈希。
        host_n, host_slots = self._match_host(seq, max_cached)
        host_n = min(host_n, max(0, self.blocks.num_free))
        if host_n > 0:
            gpu_bids = [self.blocks.allocate_fresh() for _ in range(host_n)]
            seq.block_table.extend(gpu_bids)
            job = self.offloader.start_restore(
                seq.seq_id, host_slots[:host_n], gpu_bids, kind="prefix"
            )

        for _ in range(max_cached + host_n, need):
            seq.block_table.append(self.blocks.allocate_fresh())

        seq.num_cached_tokens = (max_cached + host_n) * self.block_size
        self.last_num_cached_tokens = seq.num_cached_tokens

        if isinstance(self.index, RadixPrefixIndex) and max_cached > 0:
            self.index.pin_match(seq, max_cached)

        return self.offloader.restoring.get(seq.seq_id) if host_n > 0 else None

    def can_append(self, seq: Sequence) -> bool:
        return self.blocks.can_append(seq)

    def may_append(self, seq: Sequence) -> None:
        self.blocks.may_append(seq)

    def may_append_for(self, seq: Sequence, total_tokens: int) -> None:
        """确保页表容纳 total_tokens（投机草稿预分配）。"""
        self.blocks.may_append_for(seq, total_tokens)

    def shrink(self, seq: Sequence) -> int:
        """回收尾部多余页（投机验证后的拒绝草稿）。返回释放页数。"""
        return self.blocks.shrink(seq)

    def sync_prefix(self, seq: Sequence) -> None:
        """本步写满的新块登记进前缀索引，供后来的请求命中。"""
        if self.prefix_backend == "none" or seq.num_scheduled_tokens <= 0:
            return
        prev = seq.num_cached_tokens - seq.num_scheduled_tokens
        start = prev // self.block_size
        end = min(seq.num_full_blocks(), seq.num_cached_tokens // self.block_size)
        if start >= end:
            return
        self.index.publish(seq, end, self.blocks, start=start)

    def deallocate(self, seq: Sequence, *, publish: bool = True) -> None:
        n_full = seq.num_full_blocks() if publish else 0
        if publish and n_full > 0 and self.prefix_backend != "none":
            self.index.publish(seq, n_full, self.blocks)

        self.index.release_seq(seq.seq_id)

        published = set(seq.block_table[:n_full]) if n_full > 0 else set()
        for bid in list(seq.block_table):
            keep = bid in published and self.prefix_backend != "none"
            if not keep and self.prefix_backend == "hash":
                self.index.invalidate(bid, self.blocks)
            self.blocks.release_to_cached_or_free(bid, keep_cached=keep)
            if not keep and self.prefix_backend == "radix":
                self.index.invalidate(bid, self.blocks)

        seq.block_table.clear()
        seq.num_cached_tokens = 0
        if self.offloader is not None:
            self.offloader.drop_swap(seq.seq_id)

    # ------------------------------------------------------------------ #
    # 整条序列 swap（A1 抢占）
    # ------------------------------------------------------------------ #

    def swap_out(self, seq: Sequence) -> bool:
        """整条序列换出到主机；GPU 页释放，索引摘除，页表清空。"""
        if self.offloader is None or not self.offloader.enabled:
            return False
        if not seq.block_table:
            return False
        if not self.offloader.swap_out(seq.seq_id, list(seq.block_table)):
            return False
        for bid in list(seq.block_table):
            self.blocks.release_to_cached_or_free(bid, keep_cached=False)
            if self.blocks.blocks[bid].ref_count == 0:
                self.index.invalidate(bid, self.blocks)
        seq.block_table.clear()
        return True

    def swap_in(self, seq: Sequence) -> RestoreJob | None:
        """换入：分配新 GPU 页并发起异步 H2D。失败返回 None（按重算处理）。"""
        if self.offloader is None:
            return None
        n = seq.num_blocks
        if self.blocks.num_free < n and not self._evict_for(n):
            return None
        bids = [self.blocks.allocate_fresh() for _ in range(n)]
        job = self.offloader.swap_in(seq.seq_id, bids)
        if job is None:
            for bid in bids:
                self.blocks.release_to_cached_or_free(bid, keep_cached=False)
            return None
        seq.block_table = bids
        return job

    def _evict_for(self, n: int) -> bool:
        """腾出 n 个空页：优先挤前缀缓存。"""
        while self.blocks.num_free < n and self.blocks.cached_ids:
            bid = next(iter(self.blocks.cached_ids))
            self.blocks._evict_cached(bid)
        return self.blocks.num_free >= n
