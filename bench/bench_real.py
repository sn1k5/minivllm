"""真实环境基准（3080 Ti 12G）：吞吐 / TTFT / ITL / 各模块收益对比。

用法（仓库根目录）：
    PYTHONPATH=. python bench/bench_real.py --model Qwen3-0.6B --suite all
    可选 suite: throughput prefix spec offload fp8 w8a16 strategy all

输出：控制台表格 + bench/results/real_<卡名>.json
说明：全部 eager（不开 CUDA Graph），greedy + ignore_eos，保证数据可复现。
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import Engine, EngineConfig, SamplingParams  # noqa: E402

RESULTS_DIR = ROOT / "bench" / "results"


def make_engine(model: str, **kw) -> Engine:
    cfg = dict(
        model=model,
        enforce_eager=True,
        context_len=4096,
        max_num_seqs=64,
        max_num_batched_tokens=4096,
        kvcache_block_size=256,
        prefix_backend="hash",
        gpu_memory_utilization=0.80,
        mix_prefill_decode=True,
    )
    cfg.update(kw)
    return Engine(EngineConfig(**cfg))


def rand_tokens(engine: Engine, n: int, seed: int = 0) -> list[int]:
    """伪随机 token 序列（控制 prompt 长度用）。"""
    rng = random.Random(seed)
    vocab = engine.tokenizer.vocab_size
    return [rng.randrange(2000, vocab - 2000) for _ in range(n)]


def repeat_text(engine: Engine, phrase: str, n: int) -> list[int]:
    """重复短语构造可被 n-gram 命中的 prompt。"""
    return engine.tokenizer.encode(phrase * n)


class Runner:
    """批量提交 + 逐步驱动 + 逐请求 TTFT/ITL 观测。"""

    def __init__(self, engine: Engine):
        self.engine = engine

    def run(
        self,
        prompts: list[list[int]],
        *,
        max_tokens: int = 128,
        out_len_only: bool = True,
    ) -> dict:
        sp = SamplingParams(
            temperature=0.0, max_tokens=max_tokens, ignore_eos=True
        )
        seqs = [self.engine.add_request(ids, sp) for ids in prompts]
        arrive = {s.seq_id: time.perf_counter() for s in seqs}
        first: dict[int, float] = {}
        last: dict[int, float] = {}
        itls: list[float] = []
        t0 = time.perf_counter()
        first_cached = None
        while self.engine.has_unfinished():
            self.engine.step()
            if first_cached is None:
                first_cached = self.engine.last_num_cached_tokens
            now = time.perf_counter()
            for s in seqs:
                n = s.num_completion_tokens
                if n <= 0:
                    continue
                if s.seq_id not in first:
                    first[s.seq_id] = now
                    last[s.seq_id] = now
                elif now > last[s.seq_id]:
                    itls.append(now - last[s.seq_id])
                    last[s.seq_id] = now
        wall = time.perf_counter() - t0
        ttfts = [first[s.seq_id] - arrive[s.seq_id] for s in seqs if s.seq_id in first]
        n_tokens = sum(s.num_completion_tokens for s in seqs)
        return {
            "wall_s": round(wall, 3),
            "tokens": n_tokens,
            "tok_per_s": round(n_tokens / wall, 1),
            "ttft_mean_ms": round(statistics.mean(ttfts) * 1000, 1) if ttfts else 0,
            "ttft_p99_ms": round(_p99(ttfts) * 1000, 1) if ttfts else 0,
            "itl_mean_ms": round(statistics.mean(itls) * 1000, 2) if itls else 0,
            "itl_p99_ms": round(_p99(itls) * 1000, 2) if itls else 0,
            "prefix_hit_tokens": first_cached or 0,
        }


def _p99(xs: list[float]) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * 0.99))]


def warmup(engine: Engine, n: int = 8) -> None:
    ids = rand_tokens(engine, 32, seed=123)
    sp = SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True)
    engine.add_request(ids, sp)
    while engine.has_unfinished():
        engine.step()


def bench_throughput(model: str, out: dict) -> None:
    print("\n===== 吞吐 / TTFT / ITL（eager, greedy）=====")
    eng = make_engine(model)
    warmup(eng)
    r = Runner(eng)
    rows = {}
    for conc in (1, 4, 8, 16):
        prompts = [rand_tokens(eng, 200, seed=i) for i in range(conc)]
        rows[conc] = r.run(prompts, max_tokens=128)
        print(f"conc={conc:>2}: {rows[conc]}")
    out["throughput"] = rows
    eng.destroy()


def bench_prefix(model: str, out: dict) -> None:
    print("\n===== 前缀缓存收益（2000 token 共享前缀，逐条请求）=====")
    rows = {}

    def one_round(eng: Engine, shared, suffix_n, tag):
        r = Runner(eng)
        prompt = shared + rand_tokens(eng, suffix_n, seed=hash(tag) % 1000)
        res = r.run([prompt], max_tokens=8)
        return {
            "ttft_ms": res["ttft_mean_ms"],
            "prefix_hit_tokens": res["prefix_hit_tokens"],
        }

    # cold 基线：none 后端（每轮新 shared，无命中可言）
    eng = make_engine(model, prefix_backend="none")
    warmup(eng)
    cold = [one_round(eng, rand_tokens(eng, 2000, seed=900 + i), 16, i)
            for i in range(6)]
    rows["none_cold"] = {
        "ttft_mean_ms": round(statistics.mean(x["ttft_ms"] for x in cold), 1),
        "prefill_tokens_mean": round(
            statistics.mean(x["prefix_hit_tokens"] for x in cold), 0),
    }
    eng.destroy()

    # radix：cold（每轮新 shared）vs hot（同 shared）
    eng = make_engine(model, prefix_backend="radix")
    warmup(eng)
    cold = [one_round(eng, rand_tokens(eng, 2000, seed=950 + i), 16, 100 + i)
            for i in range(6)]
    shared = rand_tokens(eng, 2000, seed=777)
    hot = [one_round(eng, shared, 16, 200 + i) for i in range(6)]
    rows["radix_cold"] = {
        "ttft_mean_ms": round(statistics.mean(x["ttft_ms"] for x in cold), 1),
        "prefill_tokens_mean": round(
            statistics.mean(x["prefix_hit_tokens"] for x in cold), 0),
    }
    rows["radix_hot"] = {
        "ttft_mean_ms": round(statistics.mean(x["ttft_ms"] for x in hot), 1),
        "prefill_tokens_mean": round(
            statistics.mean(x["prefix_hit_tokens"] for x in hot), 0),
    }
    eng.destroy()
    c = rows["none_cold"]["ttft_mean_ms"]
    h = rows["radix_hot"]["ttft_mean_ms"]
    rows["hot_ttft_reduce_pct"] = round((1 - h / c) * 100, 1)
    print(json.dumps(rows, indent=1))
    out["prefix"] = rows


def bench_spec(model: str, out: dict) -> None:
    print("\n===== n-gram 投机解码（重复模式负载）=====")
    phrase = "def helper(x):\n    return x * 2\n\n"  # 代码复读模式
    rows = {}
    for spec in ("none", "ngram"):
        kw = {} if spec == "none" else {
            "spec_algorithm": "ngram",
            "spec_num_draft_tokens": 4,
            "spec_ngram_window": 3,
        }
        eng = make_engine(model, **kw)
        warmup(eng)
        r = Runner(eng)
        prompt = repeat_text(eng, phrase, 24)
        res = r.run([prompt], max_tokens=128)
        st = eng.scheduler.stats()
        proposed = st["spec_proposed"]
        accepted = st["spec_accepted"]
        res["accept_rate"] = round(accepted / proposed, 3) if proposed else None
        rows[spec] = res
        print(f"{spec:>6}: {rows[spec]}")
        eng.destroy()
    b, s = rows["none"]["tok_per_s"], rows["ngram"]["tok_per_s"]
    rows["speedup"] = round(s / b, 2)
    out["spec"] = rows


def bench_offload(model: str, out: dict) -> None:
    print("\n===== offload/swap（12 块小池 + 2 并发互异前缀 × 2 波）=====")
    rows = {}
    for off in (False, True):
        kw = dict(num_kvcache_blocks=12)  # 2×7 块需求 > 12：必有抢占
        if off:
            kw.update(offload_enable=True, offload_host_blocks=64, prefix_backend="radix",
                      preemption_mode="auto", swap_threshold_tokens=256)
        else:
            kw.update(prefix_backend="radix")
        eng = make_engine(model, **kw)
        warmup(eng)
        # 互不相同的长前缀：无共享；第二波重发同 prompt 测主机命中
        prompts = [rand_tokens(eng, 1400, seed=400 + i) for i in range(2)]
        r = Runner(eng)
        w1 = r.run(prompts, max_tokens=160)
        st1 = eng.scheduler.stats()
        w2 = r.run(prompts, max_tokens=160)
        st2 = eng.scheduler.stats()
        agg = lambda s: {
            "preemptions": s["preemptions"],
            "swap_out": s.get("swap_out_seqs", 0),
            "swap_in": s.get("swap_in_seqs", 0),
            "offloaded": s.get("offloaded_blocks", 0),
            "host_hits": s.get("host_hits", 0),
            "restored": s.get("restored_blocks", 0),
        }
        rows["on" if off else "off"] = {
            "wave1": {**{k: w1[k] for k in ("wall_s", "tok_per_s", "ttft_mean_ms", "itl_mean_ms")}, **agg(st1)},
            "wave2_repeat": {**{k: w2[k] for k in ("wall_s", "tok_per_s", "ttft_mean_ms", "itl_mean_ms")}, **agg(st2)},
        }
        print(f"offload={'on ' if off else 'off'}: {rows['on' if off else 'off']}")
        eng.destroy()
    out["offload"] = rows


def bench_fp8(model: str, out: dict) -> None:
    print("\n===== KV cache FP8 =====")
    rows = {}
    outputs = {}
    for dtype in ("auto", "fp8"):
        eng = make_engine(model, kv_cache_dtype=dtype)
        blocks = eng.config.num_kvcache_blocks
        warmup(eng)
        r = Runner(eng)
        prompts = [rand_tokens(eng, 200, seed=i) for i in range(8)]
        res = r.run(prompts, max_tokens=128)
        res["kv_blocks"] = blocks
        # 质量对比：自然语言 prompt 贪心输出
        ids = eng.tokenizer.encode(
            "请用三点简要说明大模型推理中 KV Cache 的作用，并说明 PagedAttention 解决了什么问题。"
        )
        sp = SamplingParams(temperature=0.0, max_tokens=128, ignore_eos=True)
        seq = eng.add_request(ids, sp)
        while eng.has_unfinished():
            eng.step()
        outputs[dtype] = seq.token_ids[seq.num_prompt_tokens:]
        rows[dtype] = res
        print(f"{dtype:>5}: {rows[dtype]}")
        eng.destroy()
    a, b = outputs["auto"], outputs["fp8"]
    match = sum(1 for x, y in zip(a, b) if x == y)
    rows["token_match_pct"] = round(match / max(len(a), len(b)) * 100, 1)
    rows["kv_mem_saving_pct"] = 50.0
    out["kv_fp8"] = rows


def bench_w8a16(model: str, out: dict) -> None:
    print("\n===== W8A16 权重量化 =====")
    rows = {}
    outputs = {}
    for quant in ("none", "w8a16"):
        eng = make_engine(model, weight_quant=quant)
        mem = sum(
            p.numel() * p.element_size()
            for p in eng._client.engine_core.executor.worker.model.parameters()
        )
        warmup(eng)
        r = Runner(eng)
        prompts = [rand_tokens(eng, 200, seed=i) for i in range(8)]
        res = r.run(prompts, max_tokens=128)
        res["weight_bytes_gb"] = round(mem / 1024**3, 2)
        ids = eng.tokenizer.encode(
            "请用三点简要说明大模型推理中 KV Cache 的作用，并说明 PagedAttention 解决了什么问题。"
        )
        sp = SamplingParams(temperature=0.0, max_tokens=128, ignore_eos=True)
        seq = eng.add_request(ids, sp)
        while eng.has_unfinished():
            eng.step()
        outputs[quant] = seq.token_ids[seq.num_prompt_tokens:]
        rows[quant] = res
        print(f"{quant:>6}: {rows[quant]}")
        eng.destroy()
    a, b = outputs["none"], outputs["w8a16"]
    match = sum(1 for x, y in zip(a, b) if x == y)
    rows["token_match_pct"] = round(match / max(len(a), len(b)) * 100, 1)
    g0 = rows["none"]["weight_bytes_gb"]
    g1 = rows["w8a16"]["weight_bytes_gb"]
    rows["weight_saving_pct"] = round((1 - g1 / g0) * 100, 1)
    out["w8a16"] = rows


def bench_strategy(model: str, out: dict) -> None:
    print("\n===== 调度策略（32 块池 + 4×2400 token + 长解码，逼出抢占）=====")
    rows = {}
    # swap 抢占需要 offload；对比 recompute / swap 两种模式
    for name, kw in (
        ("recompute", dict(preemption_mode="recompute")),
        ("swap", dict(preemption_mode="swap", offload_enable=True,
                      offload_host_blocks=128, swap_threshold_tokens=256)),
    ):
        eng = make_engine(model, num_kvcache_blocks=32, prefix_backend="none", **kw)
        warmup(eng)
        r = Runner(eng)
        prompts = [rand_tokens(eng, 2400, seed=300 + i) for i in range(4)]
        res = r.run(prompts, max_tokens=400)
        st = eng.scheduler.stats()
        res["preemptions"] = st["preemptions"]
        res["swap_out"] = st.get("swap_out_seqs", 0)
        res["swap_in"] = st.get("swap_in_seqs", 0)
        rows[name] = res
        print(f"{name:>9}: {rows[name]}")
        eng.destroy()
    # 准入控制
    for adm in ("default", "conservative"):
        eng = make_engine(model, num_kvcache_blocks=32, prefix_backend="none",
                          admission_policy=adm)
        warmup(eng)
        r = Runner(eng)
        prompts = [rand_tokens(eng, 2400, seed=300 + i) for i in range(4)]
        res = r.run(prompts, max_tokens=400)
        res["preemptions"] = eng.scheduler.stats()["preemptions"]
        rows[f"admission_{adm}"] = res
        print(f"{adm:>12}: preemptions={res['preemptions']} tok/s={res['tok_per_s']}")
        eng.destroy()
    out["strategy"] = rows


SUITES = {
    "throughput": bench_throughput,
    "prefix": bench_prefix,
    "spec": bench_spec,
    "offload": bench_offload,
    "fp8": bench_fp8,
    "w8a16": bench_w8a16,
    "strategy": bench_strategy,
}


def main() -> None:
    ap = argparse.ArgumentParser(description="minivllm 真实基准")
    ap.add_argument("--model", default="Qwen3-0.6B")
    ap.add_argument("--suite", default="all",
                    help="all / " + " ".join(SUITES))
    args = ap.parse_args()
    model = str(__import__("model_hub").resolve_model_path(args.model))
    names = list(SUITES) if args.suite == "all" else args.suite.split(",")
    RESULTS_DIR.mkdir(exist_ok=True)
    out: dict = {"model": model, "gpu": _gpu_name(), "eager": True}
    for name in names:
        SUITES[name](model, out)
    path = RESULTS_DIR / f"real_{out['gpu'].replace(' ', '_')}.json"
    merged = {}
    if path.exists():  # 增量合并，保留其他套件的历史结果
        merged = json.loads(path.read_text())
    merged.update(out)
    path.write_text(json.dumps(merged, ensure_ascii=False, indent=2))
    print(f"\nsaved -> {path}")


def _gpu_name() -> str:
    import torch

    return torch.cuda.get_device_name(0)


if __name__ == "__main__":
    main()
