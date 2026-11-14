"""metrics 单测：纯 CPU，不引 torch。"""

from __future__ import annotations

from collections import deque

from src.metrics.metrics import (
    Counter,
    Gauge,
    Histogram,
    Metrics,
    collect_engine_gauges,
)


def test_counter_and_gauge():
    c = Counter("demo_total")
    c.inc()
    c.inc(2)
    assert c.value() == 3
    g = Gauge("demo_gauge")
    g.set(0.25)
    assert g.value() == 0.25
    g.set(1)
    assert g.value() == 1


def test_histogram_buckets():
    h = Histogram("demo_seconds", [0.1, 1.0, 10.0])
    h.observe(0.05)
    h.observe(2.0)
    h.observe(100.0)  # 落 +Inf
    assert h._count == 3
    assert h._sum == 102.05
    lines = h.lines("minivllm_demo_seconds")
    assert 'minivllm_demo_seconds_bucket{le="0.1"} 1' in lines
    assert 'minivllm_demo_seconds_bucket{le="1"} 1' in lines
    assert 'minivllm_demo_seconds_bucket{le="10"} 2' in lines
    assert 'minivllm_demo_seconds_bucket{le="+Inf"} 3' in lines
    assert "minivllm_demo_seconds_sum 102.05" in lines
    assert "minivllm_demo_seconds_count 3" in lines


def test_metrics_render():
    m = Metrics()
    m.request_ok.inc(3)
    m.request_abort.inc(1)
    m.prompt_tokens_total.inc(16)
    m.generation_tokens_total.inc(5)
    m.ttft_seconds.observe(0.3)
    m.num_running.set(2)
    m.kv_cache_utilization.set(0.5)
    m.spec_accept_tokens_total.inc(4)
    m.spec_proposed_total.inc(6)
    text = m.render()
    assert '# TYPE minivllm_request_total counter' in text
    assert 'minivllm_request_total{status="ok"} 3' in text
    assert 'minivllm_request_total{status="abort"} 1' in text
    assert "# TYPE minivllm_ttft_seconds histogram" in text
    assert '# TYPE minivllm_itl_seconds histogram' in text
    assert '# TYPE minivllm_e2e_latency_seconds histogram' in text
    assert '# TYPE minivllm_queue_time_seconds histogram' in text
    assert 'minivllm_ttft_seconds_bucket{le="0.5"} 1' in text
    assert 'minivllm_ttft_seconds_bucket{le="+Inf"} 1' in text
    assert "minivllm_ttft_seconds_count 1" in text
    assert "minivllm_prompt_tokens_total 16" in text
    assert "minivllm_generation_tokens_total 5" in text
    assert "minivllm_num_running 2" in text
    assert "minivllm_kv_cache_utilization 0.5" in text
    assert "minivllm_spec_accept_tokens_total 4" in text
    assert "minivllm_spec_proposed_total 6" in text
    for name in (
        "num_waiting",
        "num_swapped",
        "prefix_cache_hit_tokens_total",
        "prefix_cache_query_tokens_total",
        "preemptions_total",
    ):
        assert f"minivllm_{name} 0" in text
    assert text.endswith("\n")


class _Blocks:
    def __init__(self, used):
        self.used_ids = used


class _KV:
    def __init__(self):
        self.num_blocks_total = 10
        self.blocks = _Blocks({1, 2, 3})


class _Sched:
    def __init__(self):
        self.waiting = deque([object()])
        self.running = deque([object(), object()])
        self.swapped = []
        self.kv = _KV()


class _Engine:
    def __init__(self, sched):
        self._sched = sched

    @property
    def scheduler(self):
        return self._sched


class _BadEngine:
    @property
    def scheduler(self):
        raise NotImplementedError("dp>1")


def test_collect_engine_gauges():
    g = collect_engine_gauges(_Engine(_Sched()))
    assert g["waiting"] == 1
    assert g["running"] == 2
    assert g["swapped"] == 0
    assert g["kv_usage"] == 0.3
    assert collect_engine_gauges(_BadEngine()) == {}
