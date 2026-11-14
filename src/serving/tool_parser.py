"""Hermes 流式工具调用解析。"""

from __future__ import annotations

import json

_OPEN = "<tool_call>"
_CLOSE = "</tool_call>"


def _tag_prefix(tag: str, text: str) -> int:
    """text 尾部可能是 tag 前缀的最长长度。"""
    n = min(len(tag) - 1, len(text))
    for k in range(n, 0, -1):
        if text.endswith(tag[:k]):
            return k
    return 0


class HermesToolParser:
    """feed 增量文本，产出 content 与 tool_call 事件。"""

    def __init__(self):
        self._state = "text"  # text / tool
        self._buf = ""  # text 态缓存
        self._body = ""  # tool 态 JSON 体
        self._index = -1  # 当前工具序号
        self._name = None
        self._args_start = -1  # arguments 起始下标
        self._emitted = 0  # 已发送 arguments 字符数
        self._scanned = 0  # 状态机扫描游标
        self._depth = 0
        self._in_str = False
        self._esc = False
        self._args_done = False

    def feed(self, delta: str) -> list[dict]:
        if self._state == "tool":
            self._body += delta
        else:
            self._buf += delta
        return self._drain(final=False)

    def flush(self) -> list[dict]:
        return self._drain(final=True)

    # ------------------------------------------------------------------

    def _drain(self, final: bool) -> list[dict]:
        evs: list[dict] = []
        while True:
            if self._state == "text":
                idx = self._buf.find(_OPEN)
                if idx >= 0:
                    head = self._buf[:idx]
                    if head:
                        evs.append({"type": "content", "text": head})
                    self._buf = self._buf[idx + len(_OPEN):]
                    self._enter_tool()
                    continue
                hold = _tag_prefix(_OPEN, self._buf)
                cut = len(self._buf) - hold
                if cut > 0:
                    evs.append({"type": "content", "text": self._buf[:cut]})
                    self._buf = self._buf[cut:]
                if final and self._buf:
                    evs.append({"type": "content", "text": self._buf})
                    self._buf = ""
                break
            close = self._body.find(_CLOSE)
            if close >= 0:
                evs.extend(self._pump(close, True))
                evs.append({"type": "tool_call_end", "index": self._index})
                self._buf += self._body[close + len(_CLOSE):]
                self._reset_tool()
                continue
            hold = _tag_prefix(_CLOSE, self._body)
            evs.extend(self._pump(len(self._body) - hold, False))
            if final:
                evs.extend(self._pump(len(self._body), True))
                evs.append({"type": "tool_call_end", "index": self._index})
                self._buf = ""
                self._reset_tool()
            break
        return evs

    def _enter_tool(self) -> None:
        self._state = "tool"
        self._body = self._buf  # 残余归 JSON 体
        self._buf = ""
        self._index += 1
        self._reset_tool_fields()

    def _reset_tool(self) -> None:
        self._state = "text"
        self._body = ""
        self._reset_tool_fields()

    def _reset_tool_fields(self) -> None:
        self._name = None
        self._args_start = -1
        self._emitted = 0
        self._scanned = 0
        self._depth = 0
        self._in_str = False
        self._esc = False
        self._args_done = False

    def _pump(self, limit: int, final: bool) -> list[dict]:
        evs: list[dict] = []
        body = self._body
        if self._name is None:  # 找 name
            i = body.find('"name"')
            if i >= 0 and i < limit:
                j = body.find('"', i + 6)
                if j > i and j < limit:
                    k = body.find('"', j + 1)
                    if k > j and k < limit:
                        self._name = body[j + 1:k]
                        evs.append(
                            {"type": "tool_call_start", "index": self._index, "name": self._name}
                        )
        if self._name is not None and self._args_start < 0:  # 找 arguments
            i = body.find('"arguments"')
            if i >= 0:
                b = body.find("{", i + 11)
                if b >= 0 and b < limit:
                    self._args_start = b
                    self._scanned = b
        if self._name is not None and self._args_start >= 0 and not self._args_done:
            safe = self._scan_args(limit)
            if safe > self._args_start + self._emitted:
                start = self._args_start + self._emitted
                evs.append({"type": "tool_call_delta", "index": self._index, "args": body[start:safe]})
                self._emitted = safe - self._args_start
            if final and self._depth > 0:  # 残缺 JSON 全吐出
                start = self._args_start + self._emitted
                rest = body[start:limit]
                if rest:
                    evs.append({"type": "tool_call_delta", "index": self._index, "args": rest})
                    self._emitted = limit - self._args_start
        return evs

    def _scan_args(self, limit: int) -> int:
        """括号深度扫描，返回可安全发送的终止下标。"""
        body = self._body
        k = self._scanned
        safe = self._scanned
        while k < limit:
            c = body[k]
            if self._in_str:
                if self._esc:
                    self._esc = False
                elif c == "\\":
                    self._esc = True
                elif c == '"':
                    self._in_str = False
            elif c == '"':
                self._in_str = True
            elif c == "{":
                self._depth += 1
            elif c == "}":
                self._depth -= 1
                if self._depth == 0:
                    self._args_done = True
                    self._scanned = k + 1
                    return k + 1
            if not self._in_str and not self._esc:
                safe = k + 1
            k += 1
        self._scanned = limit
        return safe


def parse_tool_calls(full_text: str) -> tuple[str, list[dict]]:
    """非流式解析：返回剥离后的正文与工具调用列表。"""
    parts: list[str] = []
    tools: list[dict] = []
    pos = 0
    while True:
        start = full_text.find(_OPEN, pos)
        if start < 0:
            parts.append(full_text[pos:])
            break
        parts.append(full_text[pos:start])
        end = full_text.find(_CLOSE, start + len(_OPEN))
        if end < 0:
            parts.append(full_text[start:])  # 未闭合原样保留
            break
        body = full_text[start + len(_OPEN):end].strip()
        pos = end + len(_CLOSE)
        try:
            obj = json.loads(body)
            tools.append(
                {
                    "index": len(tools),
                    "name": obj["name"],
                    "arguments": obj.get("arguments", {}),
                }
            )
        except (json.JSONDecodeError, KeyError, TypeError):
            pass  # 解析失败跳过
    return "".join(parts), tools
