# D1 — n-gram 投机解码

## 目标

无 draft 模型、零训练成本的投机解码：从序列自身的历史 token 统计 n-gram，猜出候选，
一次前向批量验证。

## 设计

### 提案（`src/speculative/ngram.py`）

- `NgramProposer` 维护 `seq_id → {上下文元组: [后续 token 列表]}`（窗口 n，默认 3）。
- prompt 首次提案时全量建表；验证后 `update()` 增量喂入新 token（含与旧尾部交叉的上下文）。
- 提案贪心链式：命中 → 提案最近出现的后继 → 上下文滑动继续查，最多 `spec_num_draft_tokens` 个。

### 调度（`src/control/scheduler.py`）

- `_attach_drafts(decodes)`：本步 decode 全部 greedy（temperature=0）才启用；
  为草稿**预分配页**（`may_append_for`），草稿表随 `ScheduledBatch` 下发。
- `_update_spec`：验证前向返回每个 q 位置的 argmax；
  与草稿逐位比对取最长接受前缀，提交 `accepted + 1` 个 token，
  `num_scheduled_tokens` 按真实提交量回填后 `sync_prefix`，`kv.shrink(seq)` 回收被拒草稿的页；
  eos 命中则截断并结束。

### 执行（`src/compute/model_runner.py`）

- decode 带草稿时走 `_run_spec`：复用 prefill 变长通路（q = 最后 token + 草稿，
  KV 全部写入），返回**所有** q 位置的 argmax——变长批天然支持各序列不同草稿数。
- 只支持 greedy：投机验证本质是对 target 分布取 argmax 与草稿比对；
  采样场景需要 rejection sampling，列入 ROADMAP。
- Decode CUDA Graph：spec 步绕过 CG（走变长通路），无草稿的普通步仍走 CG。

## 用法

```bash
python run.py --spec ngram --spec-draft-tokens 4 --temperature 0
```

## 验证

- 单测：`NgramProposer` 建表/提案/增量更新（构造重复模式序列断言提案内容）。
- 端到端：代码/复读类任务（n-gram 命中率高）输出与不开 spec 逐 token 一致（greedy 等价性）；
  `/metrics` 看 `spec_accept_tokens_total / spec_proposed_total` 的接受率与 decode 步数下降。

## 简历条目

> 实现 n-gram 投机解码：无需 draft 模型，从历史 token 统计生成草稿并批量验证，
> 接受的草稿与 PagedAttention 页管理联动（拒绝位即时回收），greedy 场景端到端加速。
