# ROADMAP — 未完成但值得做的模块（写简历前必读）

> 现有代码已覆盖：chunked prefill / continuous batching、PagedAttention + Radix 前缀缓存、
> KV offloading、抢占双模式、SLO 优先级、自适应预算、准入控制、n-gram 投机解码、
> W8A16 / KV FP8、TP/PP/DP/EP、MoE、OpenAI API、缓存感知路由、Prometheus 指标。
> 下面按"投入产出比"排序，每项标注：工作量、面试价值、简历写法。

## 一、第一优先（简历钩子，面试必问数据）

### 1. Benchmark 数据补测（0 新代码，纯跑数）
简历所有"XX%"都还没有实测数字，这是最紧迫的。需要：
- **吞吐 vs vLLM**：同机同模型（Qwen3-0.6B / 1.7B）同并发，跑 vLLM 官方 benchmark 脚本对拍。
  `bench/` 目录已有骨架（`bench/vs_vllm.py`）。
- TTFT / ITL 分布（`/metrics` 直方图导出或 bench 脚本统计）。
- MoE 层延迟降低幅度（NCU 前后对比）、GEMM shape 与 cuBLAS 对比（第二个项目的数字）。
- n-gram 投机的 accept rate 与端到端加速比（代码/复读负载 vs 普通对话负载）。
- W8A16 / KV FP8 的显存减半与吞吐变化。
**没有数字的条目在简历里不写百分比**（见简历当前写法），测出来再补。

### 2. bench 自动化脚本（1 天）
`bench/bench_strategies.py`：一键对比 {default, swap, priority, adaptive, conservative}
五种调度配置的吞吐/TTFT/ITL/preemption 计数，输出表格。
有了它，上面所有数字才可复现，面试官现场翻代码也能自己跑。

## 二、第二优先（把"进行中"变成"完成"）

### 3. PD 分离架构（1~2 周，简历已写"进行中"）
- Prefill 实例与 Decode 实例分开部署，KV 传输协议：
  prefill 完成时把 KV 页（block_table + token 长度）经传输通道送到 decode 实例。
- 单机可用两个进程 + 共享主机内存模拟（复用 B1 的 HostKVBackend 思路）；
  文档写明真机部署为 RDMA/NVLink 传输（对标 vLLM disagg / Mooncake）。
- 路由器加 `p`/`d` 角色标签，请求先到 prefill 池再到 decode 池。
- 简历写法：PD 分离部署方案与 KV 传输协议，prefill/decode 资源独立扩缩。

### 4. 分布式 KV pool（与 3 合并做）
- 独立 KVCache server 进程持有大块主机内存 + 链哈希索引（B1 的 KVOffloader 拆出去中心化）；
  推理实例按需 fetch/put。就是"KV 中心化存储 → PD 分离 → 路由器感知"一条主线。

### 5. 投机解码升级（3~5 天）
- 小模型 draft（Qwen3-0.6B draft → 1.7B target，同 tokenizer）：Eagle/HnS 风格。
- FlashInfer 的 spec decode（plan/run）接口接入 tree 验证。
- 采样场景的 rejection sampling（当前 D1 只支持 greedy）。
- 简历写法：投机解码支持 n-gram 与 draft 模型两种来源（注意：Qwen3 没有原生 MTP 头，
  不要写"原生 MTP 模块"）。

### 6. 结构化输出（2~3 天）
xgrammar/outlines 风格 logits mask 约束解码，API 加 `response_format`。
服务化故事的加分项。

## 三、第三优先（锦上添花）

| 模块 | 工作量 | 说明 |
| --- | --- | --- |
| NVMe 三级 KV 存储 | 3~4 天 | B1 加 SSD 层（LMCache 风格），主机内存为二级 |
| 前缀淘汰策略升级 | 2 天 | LRU → 命中频次加权（LFU/ARC），A4 的 preemption 计数做对照 |
| DP 负载感知分发 | 1~2 天 | dp>1 时 MPClient 从轮询升级为按队列长度/KV 水位分发 |
| 在线学习路由 | 3 天 | 规则打分 → ε-greedy 多臂老虎机，命中率反馈闭环（E2 的进阶） |
| Prefill CUDA Graph | 3~4 天 | 固定 chunk 桶 shape，prefill 也进图 |
| 请求取消/超时传播 | 1 天 | HTTP 断开 → 调度器终止序列，服务化工程完备度 |
| W4A16 / AWQ | 4~5 天 | 需要 4bit Triton kernel（dequant + groupwise scale），坑较多 |
| 跨实例前缀元数据同步 | 3 天 | 路由器全局前缀视图，降低探测开销 |

## 四、已知限制（面试被问到要能答）

1. **offload / swap 仅单机 tp=pp=1**：页号账本在控制面，跨进程无法同步页拷贝。
   多进程自动降级并打印提示。答法：架构上把拷贝后端抽象成接口，真机版走 RPC 到 worker。
2. **KV FP8 读取走 gather 反量化**：flash_attn 2.x 不支持 fp8 paged KV，
   单步延迟略增换容量翻倍；FA3/FlashInfer fp8 kernel 是升级路径。
3. **W8A16 前向固定 tile、未 autotune**：可读性优先；autotune 是现成的改进点。
4. **AsyncEngine 增量解码是全量 decode + diff**：长输出 O(n²)，可换 incremental detokenizer。
5. **spec 只支持 greedy**：见第 5 条升级路线。
6. **MoE 专家权重未量化**：3D 权重的 per-expert groupwise 量化是独立课题。
