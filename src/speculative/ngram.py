from __future__ import annotations

# ==============================================================================
# n-gram 投机解码提案器（D1）。
#
# 维护每条序列的 "上下文元组 → 后续 token 列表" 统计：
#   上下文取最近 n-1 个 token，值按出现顺序追加，提案取最近一次。
# 无需 draft 模型、无需训练；对重复模式（代码、模板、复读）命中率高。
# 仅支持 greedy（temperature=0）验证，调度器已做限制。
# ==============================================================================


class NgramProposer:
    def __init__(self, n: int = 3) -> None:
        if n < 1:
            raise ValueError(f"ngram n must be >= 1, got {n}")
        self.n = n
        # seq_id → {上下文元组: [后续 token, ...]}
        self.stats: dict[int, dict[tuple[int, ...], list[int]]] = {}

    def _ensure(self, seq) -> None:
        """首次提案时把 prompt 全量喂进统计。"""
        if seq.seq_id in self.stats:
            return
        st: dict[tuple[int, ...], list[int]] = {}
        ids = seq.token_ids
        for i in range(self.n - 1, len(ids)):
            ctx = tuple(ids[i - self.n + 1 : i])
            st.setdefault(ctx, []).append(ids[i])
        self.stats[seq.seq_id] = st

    def propose(self, seq, k: int) -> list[int]:
        """贪心链式提案：命中就顺着上一步的提案继续找，最多 k 个。"""
        if k <= 0 or seq.num_tokens < self.n:
            return []
        self._ensure(seq)
        st = self.stats[seq.seq_id]
        ids = seq.token_ids
        ctx = list(ids[len(ids) - self.n + 1 :])
        out: list[int] = []
        for _ in range(k):
            cand = st.get(tuple(ctx))
            if not cand:
                break
            nxt = cand[-1]
            out.append(nxt)
            if self.n > 1:
                ctx = ctx[1:] + [nxt]
        return out

    def update(self, seq, new_ids: list[int]) -> None:
        """验证后增量喂入新 token（含与旧尾部交叉的上下文）。"""
        st = self.stats.get(seq.seq_id)
        if st is None or not new_ids:
            return
        span = len(new_ids) + self.n - 1
        ids = seq.token_ids[len(seq.token_ids) - span :]
        for i in range(self.n - 1, len(ids)):
            ctx = tuple(ids[i - self.n + 1 : i])
            st.setdefault(ctx, []).append(ids[i])

    def drop(self, seq_id: int) -> None:
        self.stats.pop(seq_id, None)
