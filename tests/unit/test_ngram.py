"""NgramProposer 单测（纯 CPU）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.speculative.ngram import NgramProposer


class FakeSeq:
    def __init__(self, seq_id, token_ids):
        self.seq_id = seq_id
        self.token_ids = list(token_ids)

    @property
    def num_tokens(self):
        return len(self.token_ids)


def test_propose_repeats():
    p = NgramProposer(n=3)
    seq = FakeSeq(1, [5, 6, 7, 5, 6, 7, 5, 6])
    # 5,6 后面历史上两次都是 7
    assert p.propose(seq, 4) == [7, 5, 6, 7]


def test_propose_empty_history():
    p = NgramProposer(n=3)
    seq = FakeSeq(2, [1, 2])
    assert p.propose(seq, 4) == []


def test_update_incremental():
    p = NgramProposer(n=3)
    seq = FakeSeq(3, [1, 2, 3, 4])
    assert p.propose(seq, 2) == []
    seq.token_ids.append(5)
    p.update(seq, [5])
    # (3,4)→5 已入表：尾部再造 [3,4] 时应命中
    seq.token_ids.extend([3, 4])
    assert p.propose(seq, 1) == [5]


def test_drop():
    p = NgramProposer(n=2)
    seq = FakeSeq(4, [1, 2, 3])
    p.propose(seq, 1)
    p.drop(seq.seq_id)
    assert seq.seq_id not in p.stats


def test_k_limit():
    p = NgramProposer(n=2)
    seq = FakeSeq(5, [9, 9, 9, 9])
    assert p.propose(seq, 3) == [9, 9, 9]


if __name__ == "__main__":
    test_propose_repeats()
    test_propose_empty_history()
    test_update_incremental()
    test_drop()
    test_k_limit()
    print("ngram ok")
