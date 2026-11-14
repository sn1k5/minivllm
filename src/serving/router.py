"""多实例路由打分与选择。"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass


@dataclass
class Instance:
    url: str
    weight: float = 1.0


@dataclass
class ScoreWeights:
    prefix: float = 0.5
    queue: float = 0.3
    kv: float = 0.2


def score_instance(
    stats: dict,
    matched_tokens: int,
    prompt_tokens: int,
    weights: ScoreWeights,
) -> float:
    """score = 前缀命中率 + 队列余量 + KV 余量 的加权和。"""
    if not stats:
        return 0.0  # 无统计不给分
    max_seqs = int(stats.get("max_num_seqs") or 0)
    waiting = int(stats.get("waiting") or 0)
    kv_usage = float(stats.get("kv_usage") or 0.0)
    prefix_ratio = matched_tokens / max(1, prompt_tokens)
    queue_free = 1.0 - waiting / max_seqs if max_seqs > 0 else 0.0
    return (
        weights.prefix * prefix_ratio
        + weights.queue * queue_free
        + weights.kv * (1.0 - kv_usage)
    )


class Router:
    """按打分挑实例，探测失败的实例自动跳过。"""

    def __init__(
        self,
        instances: list[Instance],
        weights: ScoreWeights | None = None,
        timeout: float = 5.0,
    ):
        self.instances = instances
        self.weights = weights or ScoreWeights()
        self.timeout = timeout
        self.last_scores: list[dict] = []

    async def probe(self, client, prompt: str) -> dict:
        """并发拉取各实例 stats 与前缀命中，返回 {index: {...}}。"""

        async def fetch(inst: Instance) -> dict:
            r = await client.get(inst.url + "/internal/stats", timeout=self.timeout)
            r.raise_for_status()
            stats = r.json()
            p = await client.post(
                inst.url + "/internal/prefix_probe",
                json={"prompt": prompt},
                timeout=self.timeout,
            )
            p.raise_for_status()
            return {"stats": stats, "matched": int(p.json().get("matched_tokens", 0))}

        results = await asyncio.gather(
            *(fetch(i) for i in self.instances), return_exceptions=True
        )
        out: dict[int, dict] = {}
        for i, res in enumerate(results):
            if isinstance(res, BaseException):
                continue  # 失败实例跳过
            out[i] = res
        return out

    async def pick(self, client, prompt: str, prompt_tokens: int) -> Instance | None:
        probes = await self.probe(client, prompt)
        best: Instance | None = None
        best_score = float("-inf")
        detail: list[dict] = []
        for idx in sorted(probes):
            inst = self.instances[idx]
            data = probes[idx]
            s = score_instance(data["stats"], data["matched"], prompt_tokens, self.weights)
            s *= inst.weight
            detail.append(
                {
                    "url": inst.url,
                    "score": round(s, 6),
                    "matched": data["matched"],
                    "waiting": data["stats"].get("waiting"),
                    "kv_usage": data["stats"].get("kv_usage"),
                }
            )
            if s > best_score:
                best_score = s
                best = inst
        if detail:
            self.last_scores = detail
        return best
