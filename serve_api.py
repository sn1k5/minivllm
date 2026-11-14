"""启动 minivllm OpenAI 兼容 API 服务。

用法:
    python serve_api.py --config config.yaml --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import yaml  # noqa: E402

from model_hub import resolve_model_path  # noqa: E402
from src.config import EngineConfig  # noqa: E402


def load_engine_config(path: Path) -> EngineConfig:
    """读 yaml 并映射到 EngineConfig，缺省项走默认值。"""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        raise SystemExit(f"配置文件不存在: {path}")
    except yaml.YAMLError as exc:
        raise SystemExit(f"配置文件解析失败: {exc}")
    if not isinstance(raw, dict):
        raise SystemExit(f"配置文件应为键值映射: {path}")
    known = {f.name for f in fields(EngineConfig)} - {"hf_config", "is_moe"}
    kwargs = {k: v for k, v in raw.items() if k in known}
    name = str(kwargs.get("model") or "Qwen3-0.6B")
    kwargs["model"] = str(resolve_model_path(name))
    try:
        return EngineConfig(**kwargs)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"配置校验失败: {exc}")


def main() -> None:
    p = argparse.ArgumentParser(description="minivllm API 服务")
    p.add_argument("--config", default="config.yaml", help="yaml 配置路径")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    try:
        import uvicorn

        from src.serving.api_server import create_app
    except ImportError as exc:
        raise SystemExit(f"缺少服务依赖，先 pip install fastapi uvicorn: {exc}")

    cfg = load_engine_config(Path(args.config))
    app = create_app(cfg)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
