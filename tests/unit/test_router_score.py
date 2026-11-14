"""router 打分单测：纯 CPU，不引 torch。"""

from __future__ import annotations

import asyncio

import pytest

from src.serving.router import Instance, Router, ScoreWeights, score_instance

W = ScoreWeights()


def test_score_basic():
    stats = {"waiting": 0, "max_num_seqs": 10, "kv_usage": 0.5}
    s = score_instance(stats, 5, 10, W)
    assert s == pytest.approx(0.5 * 0.5 + 0.3 * 1.0 + 0.2 * 0.5)


def test_score_full_pressure_zero():
    stats = {"waiting": 10, "max_num_seqs": 10, "kv_usage": 1.0}
    assert score_instance(stats, 0, 10, W) == pytest.approx(0.0)


def test_score_empty_stats_zero():
    assert score_instance({}, 0, 10, W) == 0.0


def test_score_missing_fields():
    # max_num_seqs 缺 → 队列项按 0；kv 缺 → 余量满
    s = score_instance({"waiting": 3}, 0, 10, W)
    assert s == pytest.approx(0.2)


def test_score_prefix_ratio_floor():
    stats = {"waiting": 0, "max_num_seqs": 4, "kv_usage": 0.0}
    s = score_instance(stats, 8, 8, W)  # 全命中
    assert s == pytest.approx(0.5 + 0.3 + 0.2)


class _Resp:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


class _Client:
    def __init__(self, table):
        self._table = table

    def _base(self, url):
        return url.rsplit("/internal/", 1)[0]

    async def get(self, url, **kw):
        return _Resp(self._table[self._base(url)]["stats"])

    async def post(self, url, **kw):
        return _Resp({"matched_tokens": self._table[self._base(url)]["matched"]})


def test_router_pick_prefers_higher_score():
    r = Router(
        [Instance("http://a"), Instance("http://b", weight=1.0)],
        ScoreWeights(),
        timeout=1.0,
    )
    client = _Client(
        {
            "http://a": {
                "stats": {"waiting": 0, "max_num_seqs": 10, "kv_usage": 0.9},
                "matched": 0,
            },
            "http://b": {
                "stats": {"waiting": 0, "max_num_seqs": 10, "kv_usage": 0.1},
                "matched": 0,
            },
        }
    )
    inst = asyncio.run(r.pick(client, "hello", 5))
    assert inst is not None and inst.url == "http://b"
    assert len(r.last_scores) == 2


def test_router_pick_weight_boosts():
    r = Router(
        [Instance("http://a", weight=2.0), Instance("http://b")],
        ScoreWeights(prefix=1.0, queue=0.0, kv=0.0),
        timeout=1.0,
    )
    client = _Client(
        {
            "http://a": {
                "stats": {"waiting": 0, "max_num_seqs": 10, "kv_usage": 0.0},
                "matched": 4,
            },
            "http://b": {
                "stats": {"waiting": 0, "max_num_seqs": 10, "kv_usage": 0.0},
                "matched": 4,
            },
        }
    )
    inst = asyncio.run(r.pick(client, "hi", 8))
    assert inst is not None and inst.url == "http://a"


def test_router_pick_all_down():
    class _DeadClient:
        async def get(self, url, **kw):
            raise ConnectionError("down")

        async def post(self, url, **kw):
            raise ConnectionError("down")

    r = Router([Instance("http://a")], ScoreWeights(), timeout=1.0)
    assert asyncio.run(r.pick(_DeadClient(), "hi", 4)) is None
