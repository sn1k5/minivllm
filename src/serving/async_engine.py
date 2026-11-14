"""异步引擎：后台线程驱动 step，事件推给 asyncio.Queue。"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

from ..metrics.metrics import Metrics
from ..sampling_params import SamplingParams


@dataclass
class ReqState:
    """单个流式请求的跟踪状态。"""

    seq: object  # Sequence，避免引重依赖
    out_q: asyncio.Queue
    loop: asyncio.AbstractEventLoop
    stop: list[str] | None
    arrival: float
    emitted: int = 0  # 已解码 token 数
    text: str = ""  # 已产出的完整文本
    first_ts: float | None = None
    last_ts: float = 0.0
    finished: bool = False


class AsyncEngine:
    """把同步 Engine 包成异步流式接口。"""

    def __init__(self, engine, metrics: Metrics):
        self.engine = engine
        self.metrics = metrics
        self._active: dict[int, ReqState] = {}
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_evt = threading.Event()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run, name="minivllm-step", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    async def generate(
        self,
        prompt_ids: list[int],
        sp: SamplingParams,
        request_id: str,
        stop: list[str] | None = None,
    ) -> AsyncIterator[dict]:
        """流式产出 token / done 事件，提前关闭视为 abort。"""
        seq = self.engine.add_request(prompt_ids, sp)
        st = ReqState(
            seq=seq,
            out_q=asyncio.Queue(),
            loop=asyncio.get_running_loop(),
            stop=stop,
            arrival=time.perf_counter(),
        )
        m = self.metrics
        m.prompt_tokens_total.inc(seq.num_prompt_tokens)
        m.prefix_cache_query_tokens_total.inc(seq.num_prompt_tokens)
        with self._lock:
            self._active[seq.seq_id] = st
        try:
            while True:
                ev = await st.out_q.get()
                yield ev
                if ev["type"] == "done":
                    return
        finally:
            self._abort(st)

    # ------------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop_evt.is_set():
            busy = self._step_once()
            if not busy:
                self._stop_evt.wait(0.005)  # 空闲小睡

    def _step_once(self) -> bool:
        with self._lock:
            states = list(self._active.values())
        if not states and not self.engine.has_unfinished():
            return False
        try:
            finished, _ = self.engine.step()
        except Exception:
            self._abort_all()
            return True
        now = time.perf_counter()
        for st in states:
            if not st.finished:
                self._collect(st, now)
        for seq_id, _ in finished:
            with self._lock:
                st = self._active.get(seq_id)
            if st is not None and not st.finished:
                over = st.seq.num_completion_tokens >= st.seq.max_tokens
                self._finish(st, "length" if over else "stop", now)
        return True

    def _collect(self, st: ReqState, now: float) -> None:
        seq = st.seq
        n = seq.num_completion_tokens
        if n <= st.emitted:
            return
        m = self.metrics
        if st.first_ts is None:  # 首 token
            st.first_ts = now
            st.last_ts = now
            m.ttft_seconds.observe(now - st.arrival)
            m.queue_time_seconds.observe(now - st.arrival)  # 近似含排队
            try:
                m.prefix_cache_hit_tokens_total.inc(seq.num_cached_tokens)
            except Exception:
                pass
        m.generation_tokens_total.inc(n - st.emitted)
        if st.emitted > 0:
            m.itl_seconds.observe(now - st.last_ts)
        full = self.engine.tokenizer.decode(seq.token_ids[seq.num_prompt_tokens:])
        st.emitted = n
        st.last_ts = now
        cut = self._stop_cut(full, st.stop)
        if cut is not None:  # 命中停止串
            delta = cut[len(st.text):]
            if delta:
                self._emit(st, {"type": "token", "text": delta})
            self._finish(st, "stop", now, text=cut)
            return
        delta = full[len(st.text):] if full.startswith(st.text) else full
        st.text = full
        if delta:
            self._emit(st, {"type": "token", "text": delta})

    @staticmethod
    def _stop_cut(text: str, stops: list[str] | None) -> str | None:
        if not stops:
            return None
        best = -1
        for s in stops:
            if s:
                i = text.find(s)
                if i >= 0 and (best < 0 or i < best):
                    best = i
        if best < 0:
            return None
        return text[:best]

    def _finish(self, st: ReqState, reason: str, now: float, text: str | None = None) -> None:
        st.finished = True
        with self._lock:
            self._active.pop(st.seq.seq_id, None)
        self.metrics.e2e_latency_seconds.observe(now - st.arrival)
        t = st.text if text is None else text
        self._emit(st, {"type": "done", "finish_reason": reason, "text": t})

    def _abort(self, st: ReqState) -> None:
        if st.finished:
            return
        st.finished = True
        with self._lock:
            self._active.pop(st.seq.seq_id, None)
        self.metrics.e2e_latency_seconds.observe(time.perf_counter() - st.arrival)
        self._emit(st, {"type": "done", "finish_reason": "abort", "text": st.text})

    def _abort_all(self) -> None:
        with self._lock:
            states = list(self._active.values())
        for st in states:
            self._abort(st)

    @staticmethod
    def _emit(st: ReqState, ev: dict) -> None:
        try:
            st.loop.call_soon_threadsafe(st.out_q.put_nowait, ev)
        except RuntimeError:
            pass  # 事件循环已关闭
