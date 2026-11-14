"""tool_parser 单测：流式事件与非流式解析。"""

from __future__ import annotations

import json

from src.serving.tool_parser import HermesToolParser, parse_tool_calls

TOOL_TEXT = (
    "今天天气如何？<tool_call>\n"
    '{"name": "get_weather", "arguments": {"city": "北京"}}\n</tool_call>'
    "好的。<tool_call>\n"
    '{"name": "search", "arguments": {"q": "minivllm", "n": 3}}\n</tool_call>'
)

# arguments 里含转义引号与花括号的样例
ESCAPE_JSON = '{"name": "echo", "arguments": {"q": "a\\"b{]}"}}'
ESCAPE_TEXT = "前缀<tool_call>\n" + ESCAPE_JSON + "\n</tool_call>后缀"


def _feed_all(text: str, chunk: int = 7) -> list[dict]:
    p = HermesToolParser()
    evs: list[dict] = []
    for i in range(0, len(text), chunk):
        evs.extend(p.feed(text[i:i + chunk]))
    evs.extend(p.flush())
    return evs


def test_stream_two_tool_calls():
    evs = _feed_all(TOOL_TEXT)
    kinds = [e["type"] for e in evs]
    assert kinds.count("tool_call_start") == 2
    assert kinds.count("tool_call_end") == 2
    starts = [e for e in evs if e["type"] == "tool_call_start"]
    assert [s["name"] for s in starts] == ["get_weather", "search"]
    assert [s["index"] for s in starts] == [0, 1]
    ends = [e for e in evs if e["type"] == "tool_call_end"]
    assert [e["index"] for e in ends] == [0, 1]
    # content 不含标签且还原完整
    content = "".join(e["text"] for e in evs if e["type"] == "content")
    assert content == "今天天气如何？好的。"
    assert "<tool_call>" not in content and "</tool_call>" not in content
    # args 增量拼回后可解析
    args: dict[int, str] = {}
    for e in evs:
        if e["type"] == "tool_call_delta":
            args[e["index"]] = args.get(e["index"], "") + e["args"]
    assert json.loads(args[0]) == {"city": "北京"}
    assert json.loads(args[1]) == {"q": "minivllm", "n": 3}
    # start 先于本工具首个 delta
    assert kinds.index("tool_call_start") < kinds.index("tool_call_delta")


def test_args_with_escape_and_braces():
    evs = _feed_all(ESCAPE_TEXT, chunk=3)
    content = "".join(e["text"] for e in evs if e["type"] == "content")
    assert content == "前缀后缀"
    args = ""
    for e in evs:
        if e["type"] == "tool_call_delta":
            args += e["args"]
    assert json.loads(args) == {"q": 'a"b{]}'}
    _, tools = parse_tool_calls(ESCAPE_TEXT)
    assert tools[0]["arguments"] == {"q": 'a"b{]}'}


def test_incomplete_tag_kept_as_content():
    p = HermesToolParser()
    evs = p.feed("hello <tool")
    evs.extend(p.flush())
    content = "".join(e["text"] for e in evs if e["type"] == "content")
    assert content == "hello <tool"


def test_parse_tool_calls():
    content, tools = parse_tool_calls(TOOL_TEXT)
    assert content == "今天天气如何？好的。"
    assert tools[0]["name"] == "get_weather"
    assert tools[0]["arguments"] == {"city": "北京"}
    assert tools[1]["index"] == 1
    assert tools[1]["name"] == "search"
    assert tools[1]["arguments"] == {"q": "minivllm", "n": 3}


def test_parse_skips_malformed():
    text = 'hi<tool_call>\n{"name": broken\n</tool_call>bye'
    content, tools = parse_tool_calls(text)
    assert content == "hibye"
    assert tools == []
