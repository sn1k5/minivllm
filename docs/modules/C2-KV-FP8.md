# C2 — KV Cache FP8 量化

## 目标

KV cache 以 FP8（e4m3）存储，容量翻倍 → 同显存可服务并发翻倍。

## 设计

### 存储（`src/compute/kv_pool.py`）

- `kv_cache_dtype: "fp8"` 时池张量为 `float8_e4m3fn`，同时分配 per-(layer, slot) 的
  fp32 scale 表 `kv_scale`。
- `auto_num_blocks` 按fp8 字节数计算，块数自动翻倍。

### 写入（`src/compute/kv_ops.py`）

`_store_kvcache_fp8_kernel`：以该 token 的 **K 向量 amax** 定标（`scale = amax/448`），
K/V 共用同一 scale 写入，scale 落表。per-token 粒度无需校准。

### 读取（`src/compute/layers/attention.py`）

flash_attn 2.x 不直接吃 FP8 paged KV，因此 FP8 模式走
**gather 反量化 + varlen** 统一路径：
- `_gather_fp8_kv_kernel` 按页表把 FP8 页拉直成连续 KV 并乘 scale 反量化（bf16/fp16）；
- prefill：沿用 cu_seqlens 变长通路；decode：`cache_seqlens` cumsum 成 cu_k，
  q_len=1 的 varlen；CUDA Graph 兼容——`max_seqlen_k` 用 context_len 上界
  （图重放时 kernel 调度参数静态，cu_k 仍是动态张量）。

### 权衡（诚实记录）

gather 引入一次额外的 KV 读写带宽，单步 decode 延迟略增；换回的是 KV 容量翻倍。
小显存（12G 卡）上"服务更多并发"通常净收益为正，具体数值见基准计划。

## 用法

```bash
python run.py --kv-dtype fp8 --batch
```

## 验证

- 数值：同 prompt greedy 输出对比 fp16 KV，观察重合率（per-token scale 下偏差应很小）。
- 显存：KV 池 GiB 减半、可容纳 block 数翻倍。
- 性能：单并发 decode 延迟（预期略增）与高并发吞吐/可服务序列数（预期提升）。

## 简历条目

> 实现 FP8 KV cache：per-token scale 在写入内核内完成量化，读取路径按页表反量化后走
> varlen attention，与 Decode CUDA Graph 兼容，KV 显存减半、可服务并发翻倍。
