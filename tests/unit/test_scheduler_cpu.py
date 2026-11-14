"""调度器策略单测：真实 KVManager（纯 CPU，无 GPU/torch 依赖）。"""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.control.scheduler import Scheduler
from src.data import KVManager, Sequence, SequenceStatus
from src.speculative.ngram import NgramProposer

EOS = 0


def make_config(**kw):
    base = dict(
        preemption_mode="recompute",
        swap_threshold_tokens=512,
        priority_enable=False,
        adaptive_budget_enable=False,
        adaptive_budget_min=4,
        adaptive_budget_alpha=0.5,
        admission_policy="default",
        admission_gen_estimate=4,
    )
    base.update(kw)
    return types.SimpleNamespace(**base)


def make_sched(num_blocks, max_seqs=8, max_tokens=64, cfg=None):
    kv = KVManager(num_blocks=num_blocks, block_size=2, prefix_backend="hash")
    cfg = cfg or make_config()
    s = Scheduler(
        kv,
        max_num_seqs=max_seqs,
        max_num_batched_tokens=max_tokens,
        eos_token_id=EOS,
        config=cfg,
    )
    return s, kv


def test_prefill_then_decode():
    s, kv = make_sched(num_blocks=8)
    seq = Sequence.from_prompt([10, 11, 12], block_size=2)
    s.add(seq)
    sched = s.schedule()
    assert len(sched.prefills) == 1
    assert sched.prefills[0].num_scheduled_tokens == 3
    # 完成 prefill
    out = s.update(sched, [99], [])
    assert out[0].new_token_ids == [99]
    assert seq.status == SequenceStatus.RUNNING
    # decode
    sched = s.schedule()
    assert len(sched.decodes) == 1
    assert sched.decodes[0].num_scheduled_tokens == 1
    out = s.update(sched, [], [88])
    assert seq.num_tokens == 5  # 3 prompt + 99 + 88


def test_finish_on_eos():
    s, kv = make_sched(num_blocks=8)
    seq = Sequence.from_prompt([10, 11], block_size=2)
    s.add(seq)
    sched = s.schedule()
    s.update(sched, [EOS], [])
    assert seq.status == SequenceStatus.FINISHED
    assert seq.seq_id not in [x.seq_id for x in s.running]


def test_priority_order():
    cfg = make_config(priority_enable=True)
    s, kv = make_sched(num_blocks=64, max_tokens=8, cfg=cfg)
    low = Sequence.from_prompt([1, 2, 3], block_size=2)
    low.priority = 1
    high = Sequence.from_prompt([4, 5, 6], block_size=2)
    high.priority = 9
    s.add(low)
    s.add(high)
    sched = s.schedule()
    # 低优先级先入队但高优先级先被调度
    assert sched.prefills[0].seq_id == high.seq_id


def test_admission_conservative():
    cfg = make_config(admission_policy="conservative", admission_gen_estimate=8)
    s, kv = make_sched(num_blocks=4, cfg=cfg)  # 只有 4 块
    seq = Sequence.from_prompt([1, 2, 3, 4, 5, 6], block_size=2)  # 3 块 + 8 生成 ≈ 7 块 > 4
    s.add(seq)
    sched = s.schedule()
    # 兜底放行（无 decode 可跑时），不该死锁
    assert len(sched.prefills) + len(sched.decodes) >= 1 or s.is_finished()


def test_preempt_recompute():
    s, kv = make_sched(num_blocks=6)
    seqs = [Sequence.from_prompt([i, i, i, i], block_size=2) for i in range(1, 4)]
    for q in seqs:
        s.add(q)
    sched = s.schedule()
    s.update(sched, [21, 22, 23], [])
    sched = s.schedule()
    s.update(sched, [], [31, 32, 33])
    # 挤爆池：再加长请求触发抢占
    long_seq = Sequence.from_prompt(list(range(100, 120)), block_size=2)
    s.add(long_seq)
    sched = s.schedule()
    assert len(sched.prefills) == 1
    # 抢占后 running 减少
    assert len(s.running) < 3


def test_spec_flow():
    s, kv = make_sched(num_blocks=32)
    proposer = NgramProposer(n=2)
    s.set_spec(proposer, 2)
    seq = Sequence.from_prompt([7, 7, 8, 7, 7], block_size=2)
    seq.temperature = 0.0
    s.add(seq)
    sched = s.schedule()
    s.update(sched, [7], [])  # 采样 7，尾部可命中历史
    sched = s.schedule()
    assert sched.is_spec
    drafts = sched.spec_drafts[seq.seq_id]
    assert drafts == [7, 7]  # (7,) 最近延续是 7，链式重复
    d_ids = drafts + [55]  # 全接受
    out = s.update(sched, [], d_ids)
    assert out[0].new_token_ids == [7, 7, 55]
    assert seq.num_tokens == 9  # 5 prompt + 7(采样) + 草稿2 + 55(验证)


def test_spec_partial_reject():
    s, kv = make_sched(num_blocks=32)
    s.set_spec(NgramProposer(n=2), 3)
    seq = Sequence.from_prompt([7, 7, 8, 7, 7, 8, 7, 7], block_size=2)
    seq.temperature = 0.0
    s.add(seq)
    sched = s.schedule()
    s.update(sched, [7], [])
    sched = s.schedule()
    assert sched.is_spec
    drafts = sched.spec_drafts[seq.seq_id]
    assert drafts == [7, 7, 7]
    # 第一个草稿就被拒
    d_ids = [42] + [0] * 3
    out = s.update(sched, [], d_ids)
    assert out[0].new_token_ids == [42]
    assert seq.num_tokens == 10  # 8 prompt + 7 + 42


if __name__ == "__main__":
    test_prefill_then_decode()
    test_finish_on_eos()
    test_priority_order()
    test_admission_conservative()
    test_preempt_recompute()
    test_spec_flow()
    test_spec_partial_reject()
    print("scheduler ok")
