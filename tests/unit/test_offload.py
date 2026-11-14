"""KVOffloader 账本单测（fake backend，纯 CPU）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.offload import KVOffloader


class FakeBackend:
    """无 GPU 的假后端：槽位记账 + 即时就绪事件。"""

    def __init__(self, n_slots):
        self.n_slots = n_slots
        self.free = list(range(n_slots))
        self.copies = []

    def alloc_slot(self):
        return self.free.pop() if self.free else -1

    def free_slot(self, slot):
        self.free.append(slot)

    def alloc_region(self, n):
        if len(self.free) < n:
            return []
        return [self.free.pop() for _ in range(n)]

    def free_region(self, slots):
        self.free.extend(slots)

    def copy_out(self, gpu_bid, slot):
        self.copies.append(("out", gpu_bid, slot))

    def copy_out_region(self, gpu_bids, slots):
        for b, s in zip(gpu_bids, slots):
            self.copy_out(b, s)

    def copy_in_async(self, host_slots, gpu_bids):
        self.copies.append(("in", tuple(gpu_bids), tuple(host_slots)))
        return object()  # 假 event

    @staticmethod
    def event_ready(event):
        return True

    @staticmethod
    def event_synchronize(event):
        pass


def test_preserve_and_lookup():
    off = KVOffloader(host_max_blocks=4)
    off.attach_backend(FakeBackend(4))
    assert off.preserve(10, hash(1)) is True
    assert off.lookup_host(hash(1)) >= 0
    assert off.stats()["offloaded_blocks"] == 1
    assert off.stats()["host_hits"] == 1


def test_lru_evict():
    off = KVOffloader(host_max_blocks=2)
    off.attach_backend(FakeBackend(2))
    off.preserve(1, 101)
    off.preserve(2, 102)
    off.preserve(3, 103)  # 满了挤掉 101
    assert off.lookup_host(101) == -1
    assert off.lookup_host(103) >= 0


def test_restore_flow():
    off = KVOffloader(host_max_blocks=4)
    off.attach_backend(FakeBackend(4))
    job = off.start_restore(1, [0], [7], kind="prefix")
    assert off.poll_restore(1) is True
    assert 1 not in off.restoring
    assert job.n_blocks == 1


def test_swap_flow():
    off = KVOffloader(host_max_blocks=4)
    off.attach_backend(FakeBackend(4))
    assert off.swap_out(1, [1, 2, 3]) is True
    job = off.swap_in(1, [4, 5, 6])
    assert job is not None and job.n_blocks == 3
    assert off.poll_restore(1) is True
    off.drop_swap(1)  # 已 pop，无副作用


def test_no_backend():
    off = KVOffloader(host_max_blocks=4)
    assert off.enabled is False
    assert off.preserve(1, 99) is False
    assert off.swap_out(1, [1]) is False


if __name__ == "__main__":
    test_preserve_and_lookup()
    test_lru_evict()
    test_restore_flow()
    test_swap_flow()
    test_no_backend()
    print("offload ok")
