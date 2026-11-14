# minivllm

<div align="center">
  <p>
    <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.12"></a>
    <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.13+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch"></a>
    <a href="https://triton-lang.org/"><img src="https://img.shields.io/badge/Triton-3.x-76B900?style=flat-square&logo=nvidia&logoColor=white" alt="Triton"></a>
  </p>
  <p>从零实现的 Qwen3 推理引擎：调度、显存、并行、量化、投机解码、服务化全链路。</p>
</div>

---

## 特性总览

| 模块 | 能力 |
| --- | --- |
| 推理引擎核心 | chunked prefill + continuous batching、PagedAttention、hash/radix 前缀缓存 |
| KV offload | 淘汰满块换出主机内存（HBM→Host），前缀命中异步拉回，与 Radix 联动 |
| 调度策略 | recompute/swap 双模式抢占（成本模型自动选）、SLO 优先级、自适应 token 预算、保守准入 |
| 投机解码 | n-gram 提案 + 批量验证（无 draft 模型，greedy 场景） |
| 量化 | W8A16 在线权重量化（Triton dequant GEMM）、KV cache FP8 |
| 并行 | TP / PP / DP / EP（MoE 专家并行），EngineCore 多进程执行 |
| 算子 | FlashAttention paged attention、Decode CUDA Graph、fused MoE（cutlass/triton） |
| 服务化 | OpenAI 兼容 API（流式、tool calling、stop strings）、缓存感知多实例路由、Prometheus 指标 |

架构与模块设计文档：[`docs/`](docs/)（引擎原理见 `docs/00-MoE概述.md` 系列，新模块见 `docs/modules/`，待办与已知限制见 [`docs/ROADMAP-未完成与后续.md`](docs/ROADMAP-未完成与后续.md)）。

## 快速开始

```bash
# 环境（uv）
uv venv .venv --python 3.12 && source .venv/bin/activate
uv pip install torch==2.13.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
uv pip install transformers huggingface_hub safetensors tqdm numpy pyyaml einops
# 注意力后端（二选一，版本须与 torch 对齐）
uv pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.47/flash_attn-2.8.3+cu130torch2.13-cp312-cp312-linux_x86_64.whl

# 模型（首次运行会自动下载，也可手动）
hf download Qwen/Qwen3-0.6B --local-dir ~/huggingface/Qwen3-0.6B
```

```bash
# 基础推理
python run.py
# 批量推理
python run.py --batch
# 性能档（CUDA Graph + compile）
python run.py --cg --compile
# 多卡
python run.py --model Qwen3-0.6B --tp 2
```

## 新模块开关（推理侧）

```bash
python run.py --offload                      # KV offload（tp=pp=1）
python run.py --offload --preemption auto    # swap 抢占
python run.py --priority                     # SLO 优先级调度
python run.py --adaptive-budget              # 自适应 chunked prefill 预算
python run.py --admission conservative       # 保守准入
python run.py --spec ngram --temperature 0   # n-gram 投机解码
python run.py --weight-quant w8a16           # W8A16 权重量化
python run.py --kv-dtype fp8                 # KV cache FP8
```

## 服务化

```bash
uv pip install fastapi uvicorn httpx
python serve_api.py --port 8000                        # OpenAI 兼容 API + /metrics
python serve_router.py --config router.yaml --port 8100  # 多实例缓存感知路由（可选）
```

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3-0.6B","messages":[{"role":"user","content":"你好"}],"stream":true}'
```

## 代码结构

```
src/
├── control/      # Engine 门面 / EngineCore / Scheduler（调度策略）/ Executor
├── compute/      # Worker / ModelRunner / KVPool / CUDA Graph / layers / models
├── data/         # Sequence / BlockManager / KVManager / 前缀索引 / offload 账本
├── distributed/  # NCCL 并行（TP/PP）与控制通道
├── moe/          # SparseMoeBlock / 专家并行 / fused kernels
├── speculative/  # n-gram 投机解码
├── metrics/      # Prometheus 指标（自实现）
├── serving/      # OpenAI API / 路由器
└── quant 相关    # layers/w8a16.py + kv_ops FP8 内核
```

## 测试

```bash
python -m pytest tests/unit -q          # CPU 单测
python -m pytest tests/gpu -q           # GPU（需模型权重）
```

## 许可

本项目为原创学习项目，版权所有 (c) 2026 sni5s。允许个人学习、研究与教学用途的阅读与运行；
禁止以任何形式转售或用于商业用途。完整条款见 [LICENSE](LICENSE)。

> 本项目为独立实现，与 [vLLM 官方项目](https://github.com/vllm-project/vllm) 无隶属或背书关系。
> 所依赖的第三方库遵循其各自的许可证。
