"""启动多实例路由器（打分选实例并转发请求）。

用法:
    python serve_router.py --config router.yaml --port 8100
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import httpx  # noqa: E402
import uvicorn  # noqa: E402
import yaml  # noqa: E402
from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import JSONResponse, StreamingResponse  # noqa: E402

from src.serving.router import Instance, Router, ScoreWeights  # noqa: E402


def load_router_config(path: Path) -> tuple[list[Instance], ScoreWeights, float]:
    """读 router.yaml → (instances, weights, timeout)。"""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        raise SystemExit(f"配置文件不存在: {path}")
    except yaml.YAMLError as exc:
        raise SystemExit(f"配置文件解析失败: {exc}")
    if not isinstance(raw, dict):
        raise SystemExit(f"配置文件应为键值映射: {path}")
    insts = [
        Instance(url=str(i["url"]), weight=float(i.get("weight", 1.0)))
        for i in raw.get("instances") or []
        if isinstance(i, dict) and i.get("url")
    ]
    if not insts:
        raise SystemExit("router.yaml 缺少 instances 配置")
    w = raw.get("weights") or {}
    weights = ScoreWeights(
        prefix=float(w.get("prefix", 0.5)),
        queue=float(w.get("queue", 0.3)),
        kv=float(w.get("kv", 0.2)),
    )
    return insts, weights, float(raw.get("timeout", 5.0))


def create_app(insts: list[Instance], weights: ScoreWeights, timeout: float) -> FastAPI:
    router = Router(insts, weights, timeout)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.client = httpx.AsyncClient(timeout=timeout)
        yield
        await app.state.client.aclose()

    app = FastAPI(title="minivllm router", lifespan=lifespan)
    app.state.router = router

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/router/stats")
    async def router_stats():
        return {"instances": router.last_scores}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        return await _forward(request, "/v1/chat/completions")

    @app.post("/v1/completions")
    async def completions(request: Request):
        return await _forward(request, "/v1/completions")

    async def _forward(request: Request, path: str):
        client = request.app.state.client
        body = await request.body()
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            return JSONResponse({"error": "invalid json body"}, status_code=400)
        prompt = _extract_prompt(payload)
        inst = await router.pick(client, prompt, _est_tokens(prompt))
        if inst is None:
            return JSONResponse({"error": "no available instance"}, status_code=503)
        headers = {"content-type": "application/json"}
        auth = request.headers.get("authorization")
        if auth:
            headers["authorization"] = auth
        target = inst.url + path
        if payload.get("stream"):
            req = client.build_request("POST", target, content=body, headers=headers)
            resp = await client.send(req, stream=True)
            if resp.status_code != 200:
                raw = await resp.aread()
                await resp.aclose()
                return _error_response(resp.status_code, raw)
            return StreamingResponse(_relay(resp), media_type="text/event-stream")
        resp = await client.post(target, content=body, headers=headers)
        try:
            data = json.loads(resp.text)
        except json.JSONDecodeError:
            data = {"error": resp.text}
        return JSONResponse(data, status_code=resp.status_code)

    async def _relay(resp: httpx.Response):
        try:
            async for chunk in resp.aiter_bytes():
                yield chunk
        finally:
            await resp.aclose()

    def _error_response(status: int, raw: bytes):
        try:
            data = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            data = {"error": raw.decode(errors="replace")}
        return JSONResponse(data, status=status)

    return app


def _extract_prompt(payload: dict) -> str:
    """粗取 prompt 文本用于前缀探测打分。"""
    msgs = payload.get("messages")
    if isinstance(msgs, list):
        for msg in reversed(msgs):
            if not isinstance(msg, dict) or msg.get("role") != "user":
                continue
            c = msg.get("content")
            if isinstance(c, str):
                return c
            if isinstance(c, list):  # 多模态拆文本
                return " ".join(
                    str(p.get("text", "")) for p in c if isinstance(p, dict)
                )
        return ""
    prompt = payload.get("prompt", "")
    if isinstance(prompt, list):
        prompt = " ".join(str(p) for p in prompt)
    return str(prompt)


def _est_tokens(text: str) -> int:
    return max(1, len(text) // 4)  # 粗估 token 数


def main() -> None:
    p = argparse.ArgumentParser(description="minivllm 路由器")
    p.add_argument("--config", default="router.yaml", help="router.yaml 路径")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8100)
    args = p.parse_args()
    insts, weights, timeout = load_router_config(Path(args.config))
    app = create_app(insts, weights, timeout)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
