# A3 — 自适应 Chunked Prefill Token 预算

## 目标

chunked prefill 的 token 预算是定值：decode 忙时大 chunk 会挤占 decode 的 ITL，
空闲时小 chunk 又浪费算力。让预算随 decode 压力在线调整。

## 设计

- `_prefill_budget()`：`target = base * (1 - alpha * pressure)`，`pressure = running / max_num_seqs`，
  下限 `budget_min`。
- EMA 平滑（0.8/0.2）避免预算抖动引发批形状震荡。
- `_effective_budget(used)`：`min(base - used, ema_budget)`——decode 的 token 配额不受影响
  （每请求 1 token，上限仍是 max_num_seqs），只收窄 prefill chunk。
- 参数：`adaptive_budget_enable` / `adaptive_budget_min` / `adaptive_budget_alpha`。

## 用法

```bash
python run.py --adaptive-budget
```

## 验证

- 固定混合负载（长 prompt + 高并发 decode），扫描 alpha ∈ {0.3, 0.5, 0.8}，
  对比 ITL P99 与总吞吐，画出吞吐-延迟 Pareto 曲线（bench 脚本见 ROADMAP）。

## 简历条目

> chunked prefill token 预算在线自适应：按 decode 队列压力收缩 prefill chunk，
> 在吞吐与 ITL 之间取得比固定预算更优的 Pareto 前沿。
