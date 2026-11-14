# A2 — SLO 感知的优先级调度

## 目标

FCFS 对所有请求一视同仁，高优先级请求可能排在长队尾。给请求加 `priority` 与
`deadline_s`，调度器按优先级出队，deadline 临近自动提权。

## 设计

- `Sequence` 增加 `priority: int`（越大越优先）与 `deadline_s: float | None`（相对到达时刻）。
- `_prio_key(seq)`：deadline 临近/已超的请求 priority 加大权重，同级按到达顺序（seq_id）。
- `_pull_prefill_chunk` 出队顺序由 `_prio_key` 排序决定（`priority_enable` 开启时）。
- decode 侧不改：decode 每步全员推进（ITL 公平），SLO 主要作用于 TTFT。
- 请求侧：`SamplingParams(priority=.., deadline_s=..)`；OpenAI API 透传 `priority` 字段。

## 用法

```bash
python run.py --priority
# 请求：SamplingParams(temperature=0, priority=10, deadline_s=5.0)
```

## 验证

- 混合负载（高/低优先级 + 短 deadline），对比 P99/P50 TTFT 与排队时间分布（/metrics 直方图）。

## 简历条目

> 实现 SLO 感知的优先级调度：请求携带 priority/deadline，deadline 临近自动提权，
> 高优请求 P99 TTFT 显著降低。
