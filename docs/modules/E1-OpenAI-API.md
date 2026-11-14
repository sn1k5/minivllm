# E1 — OpenAI 兼容 API 服务

## 目标

把引擎暴露成 OpenAI 兼容的 HTTP 服务：chat/completions、流式 SSE、tool calling、stop strings。

## 设计

| 文件 | 职责 |
| --- | --- |
| `src/serving/async_engine.py` | 后台线程跑 `engine.step()`，把每条请求的增量 token 变成 asyncio 流 |
| `src/serving/api_server.py` | FastAPI：/v1/chat/completions、/v1/completions、/v1/models、/health、/metrics、/internal/* |
| `src/serving/tool_parser.py` | hermes 风格 `<tool_call>` 流式解析（括号深度扫描增量发 args） |
| `serve_api.py` | 启动脚本（读 config.yaml，uvicorn） |

要点：
- **AsyncEngine**：主线程做 tokenizer，工作线程 step；每步 diff 各请求的 completion token，
  全量 decode 后 diff 出增量文本（简化解码，长输出有 O(n²) 开销，标注为已知取舍）；
  stop strings 在文本层检测命中即截断。
- **chat 模板**：`apply_chat_template(messages, tools=...)` 走 Qwen3 原生模板，tools 直传。
- **流式工具调用**：HermesToolParser 把 content 与 tool_call delta 分开，
  映射成 OpenAI 的 `delta.tool_calls` 帧；非流式聚合成 `message.tool_calls`。
- **内部端点**（给路由器用）：`/internal/stats`（队列/KV 水位）、`/internal/prefix_probe`
  （前缀命中 token 数，只读探测）。
- 可选鉴权：环境变量 `MINIVLLM_API_KEY`。

## 用法

```bash
uv pip install fastapi uvicorn httpx
python serve_api.py --port 8000
curl http://127.0.0.1:8000/v1/chat/completions -d '{"model":"Qwen3-0.6B","messages":[{"role":"user","content":"你好"}]}'
```

## 验证

- 流式与非流式输出一致性；tool calling 流式帧序列合法性；
- 并发 16 请求压测下 SSE 无交错（每请求独立 asyncio.Queue）。

## 简历条目

> 提供 OpenAI 兼容 API 服务：流式输出、hermes 风格流式 tool calling 解析、stop strings，
> 多实例部署与全链路指标暴露（配合 E2/E5）。
