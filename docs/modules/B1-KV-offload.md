# B1 — KV Cache 多级 Offloading（HBM → 主机内存）

## 目标

GPU 显存有限而多轮对话的前缀大量重复。本模块把**被前缀缓存淘汰的满块**换出到主机内存留档，
后续请求命中这些前缀时再异步拉回，等效扩大 prefix cache 容量，长上下文多轮对话可服务更多并发序列。

## 设计

三个角色，职责分开：

| 角色 | 文件 | 职责 |
| --- | --- | --- |
| 账本（控制面） | `src/data/offload.py` | 主机槽位记账、链哈希→槽位二级索引、LRU 淘汰、拉回任务登记与轮询 |
| 主机池（计算面） | `src/compute/offload.py` | pinned 内存槽位（按需分配）、D2H/H2D 拷贝、独立 CUDA stream + Event |
| 淘汰挂钩 | `src/data/block_manager.py` | 页被挤出前调用 `preserve()` 留档 |

数据流：

1. **留档**：BlockManager 淘汰 cached 页时（LRU 或 radix 冷叶回收），先 `preserve(bid, hash)`——
   同步 D2H 拷到主机（调度期计算流空闲，单块约 0.5~2ms），失败则照常丢弃。
2. **命中**：`KVManager.allocate` 在 GPU radix/hash 命中断点后，用**链哈希**继续查主机索引
   （`compute_block_hash(tok_i, hash_{i-1})`），连续命中的块分配新 GPU 页并发起**异步 H2D**。
3. **等待**：调度器把该序列放入 `restoring` 队列，不占 running/waiting 名额；
   每轮 `schedule()` 先 `_poll_restores()`，Event 就绪后序列回到 waiting 正常调度——
   **拉回期间其他序列照常跑，不阻塞 step**。
4. **回收**：主机满时按 LRU 挤掉最久未用的留档页。

## 关键决策

- 链哈希做主机索引：无需第二棵 radix 树，块哈希天然携带前缀链，匹配 O(1)/块。
- 留档拷贝同步、拉回拷贝异步：写入侧在调度路径上，异步会让"页已 free 但拷贝未完成"产生竞态；
  读取侧（拉回）不阻塞任何队列，异步收益最大。
- 仅支持 `tp=pp=1`（InprocExecutor）：offload 的页号账本在控制面，跨进程无法同步页拷贝，
  多进程执行器自动关闭并打印提示。

## 用法

```bash
python run.py --offload --prefix-backend radix   # 推荐 radix，前缀复用收益最大
# config.yaml: offload_enable: true / offload_host_blocks: 512
```

## 验证与基准计划

- 单测：CPU 上验证链哈希索引与槽位 LRU（构造 fake backend）。
- GPU 冒烟：同一长 system prompt 发起 N 轮对话，`GET /metrics` 观察
  `offloaded_blocks` / `host_hits` / `restored_blocks` 增长，输出与不开 offload 一致。
- 简历数据口径：固定 prompt 集合 + 多轮对话负载，对比开/关 offload 的**可服务并发序列数**与
  prefill token 节省量。

## 简历条目（无实测数据不写数字）

> 实现 KV cache 多级 offloading（HBM → 主机内存）：换出/换入与 Radix 前缀缓存联动，
> 命中已换出前缀时异步从主机内存拉回（拉回期间不阻塞调度），长上下文多轮对话场景可服务的并发序列数提升。

## 后续

NVMe 三级存储（见 `docs/ROADMAP-未完成与后续.md`）。
