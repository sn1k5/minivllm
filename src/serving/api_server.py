"""OpenAI 兼容 API 服务。"""

from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from ..control.engine import Engine
from ..data.sequence import Sequence
from ..metrics.metrics import Metrics, collect_engine_gauges
from ..sampling_params import SamplingParams
from .async_engine import AsyncEngine
from .tool_parser import HermesToolParser, parse_tool_calls


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[dict]
    stream: bool = False
    temperature: float = 0.0
    top_p: float | None = None
    max_tokens: int = 128
    stop: list[str] | None = None
    tools: list[dict] | None = None
    priority: int = 0


class CompletionRequest(BaseModel):
    model: str | None = None
    prompt: str
    stream: bool = False
    temperature: float = 0.0
    top_p: float | None = None
    max_tokens: int = 128
    stop: list[str] | None = None
    priority: int = 0


class ProbeRequest(BaseModel):
    prompt: str


def create_app(config) -> FastAPI:
    """构建 FastAPI 应用并拉起引擎。"""
    engine = Engine(config)
    metrics = Metrics()
    async_engine = AsyncEngine(engine, metrics)
    model_name = Path(config.model).name

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async_engine.start()
        yield
        async_engine.stop()
        engine.destroy()

    app = FastAPI(title="minivllm", lifespan=lifespan)
    app.state.engine = engine
    app.state.async_engine = async_engine
    app.state.metrics = metrics
    app.state.tokenizer = engine.tokenizer
    app.state.model_name = model_name

    @app.middleware("http")
    async def _auth(request: Request, call_next):
        key = os.environ.get("MINIVLLM_API_KEY", "")
        path = request.url.path
        public = path == "/health" or path.startswith("/internal")
        if key and not public:
            if request.headers.get("authorization", "") != f"Bearer {key}":
                return JSONResponse({"error": "invalid api key"}, status_code=401)
        return await call_next(request)

    def _sse(obj: dict) -> str:
        return "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"

    def _chunk(base: dict, choice: dict) -> dict:
        return {**base, "choices": [choice]}

    def _count(done: bool) -> None:
        if done:
            metrics.request_ok.inc()
        else:
            metrics.request_abort.inc()

    def _tool_delta(base: dict, ev: dict) -> dict | None:
        kind = ev["type"]
        idx = ev.get("index", 0)
        if kind == "content":
            return _chunk(base, {"index": 0, "delta": {"content": ev["text"]}, "finish_reason": None})
        if kind == "tool_call_start":
            tc = [{"index": idx, "id": f"call_{idx}", "type": "function",
                   "function": {"name": ev["name"], "arguments": ""}}]
            return _chunk(base, {"index": 0, "delta": {"tool_calls": tc}, "finish_reason": None})
        if kind == "tool_call_delta":
            tc = [{"index": idx, "function": {"arguments": ev["args"]}}]
            return _chunk(base, {"index": 0, "delta": {"tool_calls": tc}, "finish_reason": None})
        return None  # end 事件不单独发帧

    def _openai_finish(reason: str, has_tools: bool = False) -> str:
        if has_tools:
            return "tool_calls"
        return "length" if reason == "length" else "stop"

    async def _chat_stream(req: ChatRequest, ids: list[int], sp: SamplingParams, rid: str):
        base = {"id": rid, "object": "chat.completion.chunk", "created": int(time.time()), "model": model_name}
        parser = HermesToolParser() if req.tools else None
        done = False
        finish = "stop"
        agen = async_engine.generate(ids, sp, rid, stop=req.stop)
        try:
            first = {"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}
            yield _sse(_chunk(base, first))
            async for ev in agen:
                if ev["type"] == "token":
                    if parser is None:
                        yield _sse(_chunk(base, {"index": 0, "delta": {"content": ev["text"]}, "finish_reason": None}))
                    else:
                        for te in parser.feed(ev["text"]):
                            out = _tool_delta(base, te)
                            if out is not None:
                                yield _sse(out)
                else:
                    done = True
                    finish = ev["finish_reason"]
                    if parser is not None:
                        for te in parser.flush():
                            out = _tool_delta(base, te)
                            if out is not None:
                                yield _sse(out)
                    yield _sse(_chunk(base, {"index": 0, "delta": {}, "finish_reason": _openai_finish(finish)}))
            yield "data: [DONE]\n\n"
        finally:
            _count(done)
            await agen.aclose()

    async def _chat_full(req: ChatRequest, ids: list[int], sp: SamplingParams, rid: str):
        text = ""
        finish = "stop"
        done = False
        agen = async_engine.generate(ids, sp, rid, stop=req.stop)
        try:
            async for ev in agen:
                if ev["type"] == "done":
                    done = True
                    text = ev["text"]
                    finish = ev["finish_reason"]
        finally:
            _count(done)
            await agen.aclose()
        content, tools = parse_tool_calls(text) if req.tools else (text, [])
        message = {"role": "assistant", "content": content}
        if tools:
            message["tool_calls"] = [
                {
                    "id": f"call_{t['index']}",
                    "type": "function",
                    "function": {"name": t["name"], "arguments": json.dumps(t["arguments"], ensure_ascii=False)},
                }
                for t in tools
            ]
        return JSONResponse(
            {
                "id": rid,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": [{"index": 0, "message": message, "finish_reason": _openai_finish(finish, bool(tools))}],
            }
        )

    async def _completion_stream(req: CompletionRequest, ids: list[int], sp: SamplingParams, rid: str):
        base = {"id": rid, "object": "text_completion.chunk", "created": int(time.time()), "model": model_name}
        done = False
        agen = async_engine.generate(ids, sp, rid, stop=req.stop)
        try:
            async for ev in agen:
                if ev["type"] == "token":
                    yield _sse(_chunk(base, {"index": 0, "text": ev["text"], "finish_reason": None}))
                else:
                    done = True
                    yield _sse(_chunk(base, {"index": 0, "text": "", "finish_reason": _openai_finish(ev["finish_reason"])}))
            yield "data: [DONE]\n\n"
        finally:
            _count(done)
            await agen.aclose()

    async def _completion_full(req: CompletionRequest, ids: list[int], sp: SamplingParams, rid: str):
        text = ""
        finish = "stop"
        done = False
        agen = async_engine.generate(ids, sp, rid, stop=req.stop)
        try:
            async for ev in agen:
                if ev["type"] == "done":
                    done = True
                    text = ev["text"]
                    finish = ev["finish_reason"]
        finally:
            _count(done)
            await agen.aclose()
        return JSONResponse(
            {
                "id": rid,
                "object": "text_completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": [{"index": 0, "text": text, "finish_reason": _openai_finish(finish)}],
            }
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatRequest):
        prompt = engine.tokenizer.apply_chat_template(
            req.messages, tokenize=False, add_generation_prompt=True, tools=req.tools or None
        )
        ids = engine.tokenizer.encode(prompt)
        sp = SamplingParams(
            temperature=req.temperature, top_p=req.top_p, max_tokens=req.max_tokens, priority=req.priority
        )
        rid = "chatcmpl-" + uuid.uuid4().hex
        if req.stream:
            return StreamingResponse(_chat_stream(req, ids, sp, rid), media_type="text/event-stream")
        return await _chat_full(req, ids, sp, rid)

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest):
        ids = engine.tokenizer.encode(req.prompt)
        sp = SamplingParams(
            temperature=req.temperature, top_p=req.top_p, max_tokens=req.max_tokens, priority=req.priority
        )
        rid = "cmpl-" + uuid.uuid4().hex
        if req.stream:
            return StreamingResponse(_completion_stream(req, ids, sp, rid), media_type="text/event-stream")
        return await _completion_full(req, ids, sp, rid)

    @app.get("/v1/models")
    async def list_models():
        return {"object": "list", "data": [{"id": model_name, "object": "model"}]}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/metrics")
    async def metrics_endpoint():
        _update_gauges()
        return Response(media_type="text/plain", content=metrics.render())

    @app.get("/internal/stats")
    async def internal_stats():
        g = collect_engine_gauges(engine)
        return {
            "model": model_name,
            "waiting": g.get("waiting", 0),
            "running": g.get("running", 0),
            "swapped": g.get("swapped", 0),
            "kv_usage": g.get("kv_usage", 0.0),
            "max_num_seqs": config.max_num_seqs,
            "prefix_backend": config.prefix_backend,
        }

    @app.post("/internal/prefix_probe")
    async def prefix_probe(req: ProbeRequest):
        ids = engine.tokenizer.encode(req.prompt)
        try:
            seq = Sequence.from_prompt(ids, block_size=engine.block_size)
            matched = engine.scheduler.kv.probe_prefix_len(seq) * engine.block_size
        except Exception:
            matched = 0
        return {"matched_tokens": matched}

    def _update_gauges() -> None:
        g = collect_engine_gauges(engine)
        if not g:
            return
        metrics.num_waiting.set(g["waiting"])
        metrics.num_running.set(g["running"])
        metrics.num_swapped.set(g["swapped"])
        metrics.kv_cache_utilization.set(g["kv_usage"])

    return app
