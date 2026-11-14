"""自实现 Prometheus 指标，零第三方依赖。"""

from __future__ import annotations


def _fmt(v: float) -> str:
    return f"{v:g}"


def _labels_suffix(labels: dict) -> str:
    if not labels:
        return ""
    items = [f'{k}="{v}"' for k, v in sorted(labels.items())]
    return "{" + ",".join(items) + "}"


class Counter:
    """只增计数器。"""

    def __init__(self, name: str, labels: dict | None = None):
        self.name = name
        self.labels = labels or {}
        self._value = 0.0

    def inc(self, n: float = 1.0) -> None:
        self._value += n

    def value(self) -> float:
        return self._value


class Gauge:
    """可增可减的瞬时值。"""

    def __init__(self, name: str):
        self.name = name
        self._value = 0.0

    def set(self, v: float) -> None:
        self._value = v

    def value(self) -> float:
        return self._value


class Histogram:
    """固定桶直方图，桶内独立计数，渲染时累计。"""

    def __init__(self, name: str, buckets: list[float]):
        self.name = name
        self.buckets = sorted(buckets)
        self._counts = [0] * len(self.buckets)  # 各桶独立计数
        self._count = 0
        self._sum = 0.0

    def observe(self, v: float) -> None:
        self._count += 1
        self._sum += v
        for i, ub in enumerate(self.buckets):
            if v <= ub:
                self._counts[i] += 1
                return
        # 超出最大桶只进 +Inf

    def lines(self, metric: str) -> list[str]:
        out: list[str] = []
        cum = 0
        for i, ub in enumerate(self.buckets):
            cum += self._counts[i]
            out.append(f'{metric}_bucket{{le="{_fmt(ub)}"}} {cum}')
        out.append(f'{metric}_bucket{{le="+Inf"}} {self._count}')
        out.append(f"{metric}_sum {_fmt(self._sum)}")
        out.append(f"{metric}_count {self._count}")
        return out


_TTFT_BUCKETS = [0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10]
_ITL_BUCKETS = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1]


class Metrics:
    """预建全部指标，render 输出文本格式。"""

    def __init__(self):
        self.request_ok = Counter("request_total", {"status": "ok"})
        self.request_abort = Counter("request_total", {"status": "abort"})
        self.prompt_tokens_total = Counter("prompt_tokens_total")
        self.generation_tokens_total = Counter("generation_tokens_total")
        self.ttft_seconds = Histogram("ttft_seconds", _TTFT_BUCKETS)
        self.itl_seconds = Histogram("itl_seconds", _ITL_BUCKETS)
        self.e2e_latency_seconds = Histogram("e2e_latency_seconds", _TTFT_BUCKETS)
        self.queue_time_seconds = Histogram("queue_time_seconds", _TTFT_BUCKETS)
        self.num_waiting = Gauge("num_waiting")
        self.num_running = Gauge("num_running")
        self.num_swapped = Gauge("num_swapped")
        self.kv_cache_utilization = Gauge("kv_cache_utilization")
        self.prefix_cache_hit_tokens_total = Counter("prefix_cache_hit_tokens_total")
        self.prefix_cache_query_tokens_total = Counter("prefix_cache_query_tokens_total")
        self.preemptions_total = Counter("preemptions_total")
        self.spec_accept_tokens_total = Counter("spec_accept_tokens_total")
        self.spec_proposed_total = Counter("spec_proposed_total")
        self._registry: list[tuple[str, object]] = [
            ("counter", self.request_ok),
            ("counter", self.request_abort),
            ("counter", self.prompt_tokens_total),
            ("counter", self.generation_tokens_total),
            ("histogram", self.ttft_seconds),
            ("histogram", self.itl_seconds),
            ("histogram", self.e2e_latency_seconds),
            ("histogram", self.queue_time_seconds),
            ("gauge", self.num_waiting),
            ("gauge", self.num_running),
            ("gauge", self.num_swapped),
            ("gauge", self.kv_cache_utilization),
            ("counter", self.prefix_cache_hit_tokens_total),
            ("counter", self.prefix_cache_query_tokens_total),
            ("counter", self.preemptions_total),
            ("counter", self.spec_accept_tokens_total),
            ("counter", self.spec_proposed_total),
        ]

    def render(self) -> str:
        lines: list[str] = []
        for kind, obj in self._registry:
            metric = f"minivllm_{obj.name}"
            lines.append(f"# TYPE {metric} {kind}")
            if kind == "histogram":
                lines.extend(obj.lines(metric))
            else:
                lines.append(f"{metric}{_labels_suffix(getattr(obj, 'labels', {}))} {_fmt(obj.value())}")
        return "\n".join(lines) + "\n"


def collect_engine_gauges(engine) -> dict:
    """读调度器队列与 KV 占用，dp>1 或异常时返回空 dict。"""
    try:
        sched = engine.scheduler
        total = int(sched.kv.num_blocks_total)
        used = len(sched.kv.blocks.used_ids)
        return {
            "waiting": len(sched.waiting),
            "running": len(sched.running),
            "swapped": len(sched.swapped),
            "kv_usage": used / total if total > 0 else 0.0,
        }
    except Exception:
        return {}
