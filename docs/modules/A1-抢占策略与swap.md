# A1 — 抢占策略：recompute / swap 双模式

## 目标

显存不足时必须抢占。原有实现只有 recompute（释放全部 KV、序列回 waiting 重算），
长序列被抢占时重算成本极高。本模块增加 **swap 模式**：整条序列的 KV 换出到主机，
恢复时换回，不重算。

## 设计

- `src/control/scheduler.py`：`preempt(seq)` 按 `_preempt_mode(seq)` 分派。
  - `recompute`：原有路径（`deallocate` + 回 waiting 队首）。
  - `swap`：`KVManager.swap_out(seq)` → 全部页 D2H 到主机连续区域 → GPU 页释放、索引摘除 →
    序列进 `swapped` 队列；`_try_swap_in()` 每轮调度页够时发起异步 H2D（复用 B1 的拉回通道），
    Event 就绪后回 running（decode 不变量 `len == cached+1` 天然保持）。
  - `auto`（默认）：成本模型——已计算 KV 的 token 数 `num_cached_tokens >= swap_threshold_tokens`
    则 swap（重算太亏），否则 recompute（换出换入本身也要带宽，短序列不划算）。
- swap 失败（主机区域不够）自动回退 recompute。

## 用法

```bash
python run.py --offload --preemption auto     # swap 依赖 offload 的主机池
# config.yaml: preemption_mode: "auto" / swap_threshold_tokens: 512
```

## 验证

- 构造 `max_num_seqs` 大、`num_kvcache_blocks` 小的压力负载，迫使抢占；
  `/metrics` 观察 `swap_out_seqs` / `swap_in_seqs`，对比三种模式的吞吐与 preemption 次数。
- 长序列（已算 > threshold）场景 swap 模式应显著减少重复 prefill。

## 简历条目

> 设计序列抢占策略：recompute 与 swap 双模式，按被抢占序列已计算的 KV 量做成本模型自动选择，
> 长序列场景避免重复 prefill。
