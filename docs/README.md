# 技术文档

| 文档 | 写什么 |
| --- | --- |
| [00-MoE概述.md](00-MoE概述.md) | 从 Jacobs 到 Qwen3-MoE：稀疏门控、接到 FFN、Switch、Mixtral、DeepSeek、Qwen 系列；常考题 |
| [01-路由与融合计算.md](01-路由与融合计算.md) | 从 naive 双重循环到 align / grouped GEMM，再串回本仓库前向 |
| [02-专家并行.md](02-专家并行.md) | MoE TP 与 EP、token 不搬家的 EP、共享专家与 AR 顺序、常考题 |
| [03-MoE实现.md](03-MoE实现.md) | `src/moe/` 文件地图、权重加载、后端、与官方差异、常考题 |

新模块（调度策略 / KV offload / 投机解码 / 量化 / 服务化）的设计文档在 `docs/modules/`。

代码入口：[`src/moe/`](../src/moe/)。模型：[`src/compute/models/qwen3_moe.py`](../src/compute/models/qwen3_moe.py)。测试默认模型见 [`tests/README.md`](../tests/README.md)：Qwen1.5-MoE-A2.7B 跑 TP/EP，Qwen3-30B-A3B 做大模型冒烟。

配图：`posts/` 里 `hf_` / `mixtral_` / `deepseek_` / `switch_` 前缀为论文或博客原图，可用 `bash docs/posts/fetch_moe_figs.sh` 重拉。`dense_vs_sparse.png` 一类为本实现示意图，用 `python docs/posts/gen_moe_figs.py` 生成。
