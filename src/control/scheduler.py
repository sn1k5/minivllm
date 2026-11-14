from __future__ import annotations

from collections import deque

from ..data import EngineCoreOutput, KVManager, ScheduledBatch, Sequence, SequenceStatus

# ==============================================================================
# 请求调度器（并行无关）。
#
# 只管理 waiting / running / swapped / restoring 四个队列与 KV 页预算，
# 产出 ScheduledBatch。TP/PP/DP 均不在此体现：同一份 schedule 由 Executor
# 广播（TP/PP）或由各 DP 副本各自独立调度（DP）。
#
# 调度策略（均可用配置开关）：
#   A1 抢占：recompute / swap 双模式 + auto 成本选择（swap 需 offload）
#   A2 SLO：优先级队列（priority 越大越优先，deadline 临近自动提权）
#   A3 自适应 token 预算：decode 压力大时收窄 prefill chunk
#   A4 准入控制：conservative 模式预估整条生命周期的页需求
#   D1 投机解码：decode 步附加 n-gram 草稿，批量验证
# ==============================================================================


class Scheduler:
    def __init__(
        self,
        kv: KVManager,
        *,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        eos_token_id: int,
        mix_prefill_decode: bool = True,
        config=None,
    ) -> None:
        self.kv = kv
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.eos = eos_token_id
        self.mix_prefill_decode = mix_prefill_decode
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.swapped: list[Sequence] = []  # A1 换出待换入
        self.restoring: list[Sequence] = []  # B1/A1 异步拉回中

        cfg = config
        # A1
        self.preemption_mode = getattr(cfg, "preemption_mode", "recompute")
        self.swap_threshold = getattr(cfg, "swap_threshold_tokens", 512)
        # A2
        self.priority_enable = getattr(cfg, "priority_enable", False)
        # A3
        self.adaptive_budget = getattr(cfg, "adaptive_budget_enable", False)
        self.budget_min = getattr(cfg, "adaptive_budget_min", 512)
        self.budget_alpha = getattr(cfg, "adaptive_budget_alpha", 0.5)
        self.budget_ema = float(max_num_batched_tokens)
        # A4
        self.admission = getattr(cfg, "admission_policy", "default")
        self.gen_estimate = getattr(cfg, "admission_gen_estimate", 256)
        # D1
        self.spec_proposer = None
        self.spec_k = 0
        self.spec_drafts: dict[int, list[int]] = {}
        # 观测：抢占计数（A1/A4 效果对比用）
        self.n_preempts = 0
        # 观测：投机提案/接受 token 数（D1 接受率用）
        self.n_spec_proposed = 0
        self.n_spec_accepted = 0

    # ------------------------------------------------------------------ #
    # 队列基础
    # ------------------------------------------------------------------ #

    def is_finished(self) -> bool:
        return (
            not self.waiting
            and not self.running
            and not self.swapped
            and not self.restoring
        )

    def add(self, seq: Sequence) -> None:
        seq.status = SequenceStatus.WAITING
        self.waiting.append(seq)

    def set_spec(self, proposer, num_draft_tokens: int) -> None:
        """启用投机解码（EngineCore 在 spec_algorithm != none 时调用）。"""
        self.spec_proposer = proposer
        self.spec_k = max(0, num_draft_tokens)

    def stats(self) -> dict:
        """调度器统计快照（bench 与指标用）。"""
        out = {
            "num_waiting": len(self.waiting),
            "num_running": len(self.running),
            "num_swapped": len(self.swapped),
            "num_restoring": len(self.restoring),
            "preemptions": self.n_preempts,
            "spec_proposed": self.n_spec_proposed,
            "spec_accepted": self.n_spec_accepted,
        }
        if self.kv.offloader is not None:
            out.update(self.kv.offloader.stats())
        return out

    # ------------------------------------------------------------------ #
    # 主调度入口
    # ------------------------------------------------------------------ #

    def schedule(self) -> ScheduledBatch:
        self._poll_restores()
        self._try_swap_in()

        if self.mix_prefill_decode:
            prefills, decodes = self._schedule_mixed()
        else:
            prefills, decodes = self._schedule_prefill_first()

        is_spec = self._attach_drafts(decodes)
        self._budget_ema_update(len(decodes))
        return ScheduledBatch(
            prefills=prefills,
            decodes=decodes,
            is_spec=is_spec,
            spec_drafts=dict(self.spec_drafts),
        )

    # ------------------------------------------------------------------ #
    # B1 / A1：异步拉回与换入
    # ------------------------------------------------------------------ #

    def _poll_restores(self) -> None:
        off = self.kv.offloader
        for seq in list(self.restoring):
            if off is None or off.poll_restore(seq.seq_id):
                self.restoring.remove(seq)
                if seq.status == SequenceStatus.SWAPPED:
                    seq.status = SequenceStatus.RUNNING
                    self.running.append(seq)
                else:
                    seq.status = SequenceStatus.WAITING
                    self.waiting.append(seq)

    def _try_swap_in(self) -> None:
        """A1：页够就发起换入（异步）；换入失败转重算。"""
        for seq in list(self.swapped):
            # cached 里的已发布前缀可被驱逐腾位，swap_in 内部会做驱逐
            avail = self.kv.num_free + len(self.kv.blocks.cached_ids)
            if avail < seq.num_blocks:
                continue
            job = self.kv.swap_in(seq)
            self.swapped.remove(seq)
            if job is None:
                seq.num_cached_tokens = 0
                seq.status = SequenceStatus.WAITING
                self.waiting.append(seq)
                continue
            self.restoring.append(seq)

    # ------------------------------------------------------------------ #
    # A1：抢占
    # ------------------------------------------------------------------ #

    def preempt(self, seq: Sequence) -> None:
        if seq.status == SequenceStatus.FINISHED:
            return
        self.n_preempts += 1
        if self._preempt_mode(seq) == "swap" and seq.block_table and self.kv.swap_out(seq):
            seq.status = SequenceStatus.SWAPPED
            self.swapped.append(seq)
            return
        seq.status = SequenceStatus.WAITING
        self.kv.deallocate(seq, publish=False)
        self.waiting.appendleft(seq)

    def _preempt_mode(self, seq: Sequence) -> str:
        if self.preemption_mode != "auto":
            return self.preemption_mode
        off = self.kv.offloader
        if off is not None and off.enabled and seq.num_cached_tokens >= self.swap_threshold:
            return "swap"  # 已算的多，换出比重算划算
        return "recompute"

    # ------------------------------------------------------------------ #
    # A2：优先级
    # ------------------------------------------------------------------ #

    def _prio_key(self, seq: Sequence):
        """排序键：deadline 临近的提权；同级按到达顺序。"""
        if not self.priority_enable:
            return (0, seq.seq_id)
        prio = seq.priority
        if seq.deadline_s is not None:
            # deadline 相对 add 时刻；超时风险大就提到最前。
            prio += 10**6
        return (-prio, seq.seq_id)

    # ------------------------------------------------------------------ #
    # A3：自适应预算
    # ------------------------------------------------------------------ #

    def _prefill_budget(self) -> int:
        if not self.adaptive_budget:
            return self.max_num_batched_tokens
        pressure = len(self.running) / max(1, self.max_num_seqs)
        target = self.max_num_batched_tokens * (1.0 - self.budget_alpha * pressure)
        return int(max(self.budget_min, target))

    def _budget_ema_update(self, n_decodes: int) -> None:
        if not self.adaptive_budget:
            return
        self.budget_ema = 0.8 * self.budget_ema + 0.2 * self._prefill_budget()
        del n_decodes

    def _effective_budget(self, tokens_used: int) -> int:
        """本步 prefill 还能用多少 token：总量封顶 + 自适应预算。"""
        base = self.max_num_batched_tokens - tokens_used
        if not self.adaptive_budget:
            return base
        budget = int(self.budget_ema)
        return max(0, min(base, budget))

    # ------------------------------------------------------------------ #
    # A4：准入
    # ------------------------------------------------------------------ #

    def _can_admit(self, seq: Sequence) -> bool:
        if self.admission != "conservative":
            return True
        total = seq.num_tokens + self.gen_estimate
        est_blocks = (total + seq.block_size - 1) // seq.block_size
        have = len(seq.block_table)
        return self.kv.blocks.available_slots + have >= est_blocks

    # ------------------------------------------------------------------ #
    # D1：n-gram 草稿
    # ------------------------------------------------------------------ #

    def _attach_drafts(self, decodes: list[Sequence]) -> bool:
        """给本步 decode 附加草稿。非全 greedy 或无提案时不启用。"""
        self.spec_drafts = {}
        if self.spec_proposer is None or self.spec_k <= 0 or not decodes:
            return False
        if any(s.temperature > 1e-5 for s in decodes):
            return False
        for seq in decodes:
            drafts = self.spec_proposer.propose(seq, self.spec_k)
            if drafts:
                self.kv.may_append_for(seq, seq.num_tokens + len(drafts))
                self.spec_drafts[seq.seq_id] = drafts
        self.n_spec_proposed += sum(len(v) for v in self.spec_drafts.values())
        return True

    # ------------------------------------------------------------------ #
    # decode 调度
    # ------------------------------------------------------------------ #

    def _room(self, prefills: list[Sequence], decodes: list[Sequence]) -> bool:
        return len(prefills) + len(decodes) < self.max_num_seqs

    def _try_alloc(self, seq: Sequence) -> bool:
        if seq.block_table:
            return True
        if not self.kv.can_allocate(seq):
            return False
        job = self.kv.allocate(seq)
        if job is not None:  # B1：主机留档拉回中，移出等待队列
            self.restoring.append(seq)
            self.waiting.remove(seq)
            return False
        return True

    def _schedule_decode(
        self,
        prefills: list[Sequence],
        decodes: list[Sequence],
        tokens_used: int,
    ) -> int:
        while self.running and self._room(prefills, decodes):
            seq = self.running.popleft()

            # 需要 (1+草稿) 个新 token 的页；不够则抢占他人或自己。
            total = seq.num_tokens + self.spec_k + 1
            bs = seq.block_size
            need = (total + bs - 1) // bs - len(seq.block_table)
            if need > 0 and self.kv.blocks.available_slots < need:
                while self.running and self.kv.blocks.available_slots < need:
                    self.preempt(self.running.pop())
                if self.kv.blocks.available_slots < need:
                    self.preempt(seq)
                    break

            seq.num_scheduled_tokens = 1
            self.kv.may_append(seq)
            decodes.append(seq)
            tokens_used += 1

        if decodes:
            self.running.extendleft(reversed(decodes))

        return tokens_used

    def _pull_prefill_chunk(
        self,
        prefills: list[Sequence],
        decodes: list[Sequence],
        tokens_used: int,
        *,
        allow_trailing_partial: bool = True,
    ) -> int:
        if not self.waiting or not self._room(prefills, decodes):
            return tokens_used

        n_partial = 0
        skipped_all = True
        ordered = (
            sorted(self.waiting, key=self._prio_key) if self.priority_enable
            else list(self.waiting)
        )
        for seq in ordered:
            if not self._room(prefills, decodes):
                break

            remaining = self._effective_budget(tokens_used)
            if remaining <= 0:
                break

            if not self._can_admit(seq):
                continue  # A4：暂缓准入，看下一条

            skipped_all = False
            if not self._try_alloc(seq):
                if seq.seq_id in {s.seq_id for s in self.restoring}:
                    continue  # B1：进入拉回，看下一条
                break  # 池真不够，停

            need = seq.num_tokens - seq.num_cached_tokens
            if need <= 0:
                seq.status = SequenceStatus.RUNNING
                self.waiting.remove(seq)
                self.running.append(seq)
                continue

            if remaining < need and prefills and not allow_trailing_partial:
                break

            chunk = min(need, remaining)
            seq.num_scheduled_tokens = chunk
            prefills.append(seq)
            tokens_used += chunk

            if seq.num_cached_tokens + chunk == seq.num_tokens:
                seq.need_sample = True
                seq.status = SequenceStatus.RUNNING
                self.waiting.remove(seq)
                self.running.append(seq)
            else:
                seq.need_sample = False
                n_partial += 1
                if n_partial >= 1:
                    break

        # A4 死锁兜底：全员被准入卡住且无活可跑时，放行优先级最高的一条。
        if skipped_all and self.waiting and not prefills and not decodes:
            seq = min(self.waiting, key=self._prio_key)
            if self._try_alloc(seq):
                need = seq.num_tokens - seq.num_cached_tokens
                if need > 0:
                    chunk = min(need, self._effective_budget(tokens_used))
                    if chunk > 0:
                        seq.num_scheduled_tokens = chunk
                        prefills.append(seq)
                        tokens_used += chunk
                        if seq.num_cached_tokens + chunk == seq.num_tokens:
                            seq.need_sample = True
                            seq.status = SequenceStatus.RUNNING
                            self.waiting.remove(seq)
                            self.running.append(seq)
        return tokens_used

    # ------------------------------------------------------------------ #
    # prefill 优先 / 混跑
    # ------------------------------------------------------------------ #

    def _schedule_prefill_first(self) -> tuple[list[Sequence], list[Sequence]]:
        prefills: list[Sequence] = []
        decodes: list[Sequence] = []
        self._pull_prefill_chunk(prefills, decodes, 0, allow_trailing_partial=False)
        if prefills:
            return prefills, []
        self._schedule_decode(prefills, decodes, 0)
        return [], decodes

    def _schedule_mixed(self) -> tuple[list[Sequence], list[Sequence]]:
        prefills: list[Sequence] = []
        decodes: list[Sequence] = []
        used = self._schedule_decode(prefills, decodes, 0)
        if decodes:
            self._pull_prefill_chunk(prefills, decodes, used)
            return prefills, decodes
        return self._schedule_prefill_first()

    # ------------------------------------------------------------------ #
    # 后处理
    # ------------------------------------------------------------------ #

    def update(
        self, sched: ScheduledBatch, p_ids: list[int], d_ids: list[int]
    ) -> list[EngineCoreOutput]:
        outputs: list[EngineCoreOutput] = []
        if sched.is_spec:
            outputs.extend(self._update_spec(sched, d_ids))
        elif sched.decodes:
            outputs.extend(self._update_batch(sched.decodes, d_ids, is_prefill=False))
        if sched.prefills:
            outputs.extend(self._update_batch(sched.prefills, p_ids, is_prefill=True))
        return outputs

    def _update_spec(
        self, sched: ScheduledBatch, d_ids: list[int]
    ) -> list[EngineCoreOutput]:
        """投机验证：贪心匹配草稿，提交接受前缀，回收被拒的页。"""
        outputs: list[EngineCoreOutput] = []
        finished: set[int] = set()
        offset = 0

        for seq in sched.decodes:
            drafts = sched.spec_drafts.get(seq.seq_id, [])
            k1 = len(drafts) + 1
            verify = d_ids[offset : offset + k1]
            offset += k1

            accepted = 0
            for i, d in enumerate(drafts):
                if verify[i] == d:
                    accepted += 1
                else:
                    break
            self.n_spec_accepted += accepted
            new_ids = drafts[:accepted] + [verify[accepted]]

            seq.num_cached_tokens += 1 + accepted
            seq.num_scheduled_tokens = 1 + accepted
            self.kv.sync_prefix(seq)
            seq.num_scheduled_tokens = 0

            for t in new_ids:
                seq.append_token(t)
            self.kv.shrink(seq)  # 回收被拒草稿的页

            finish = False
            if seq.max_tokens <= 0 or seq.num_completion_tokens >= seq.max_tokens:
                finish = True
            elif not seq.ignore_eos and self.eos in new_ids:
                idx = new_ids.index(self.eos)
                keep = seq.num_tokens - (len(new_ids) - (idx + 1))
                del seq.token_ids[keep:]
                seq.last_token = seq.token_ids[-1]
                new_ids = new_ids[: idx + 1]
                finish = True

            if finish:
                seq.status = SequenceStatus.FINISHED
                self.kv.deallocate(seq, publish=True)
                finished.add(seq.seq_id)
                if self.spec_proposer is not None:
                    self.spec_proposer.drop(seq.seq_id)

            outputs.append(
                EngineCoreOutput(
                    seq_id=seq.seq_id,
                    new_token_ids=new_ids,
                    is_finished=seq.seq_id in finished,
                )
            )

        if finished:
            self.running = deque(s for s in self.running if s.seq_id not in finished)
        return outputs

    def _update_batch(
        self,
        seqs: list[Sequence],
        token_ids: list[int],
        *,
        is_prefill: bool,
    ) -> list[EngineCoreOutput]:
        finished: set[int] = set()
        outputs: list[EngineCoreOutput] = []

        for seq, token_id in zip(seqs, token_ids):
            seq.num_cached_tokens += seq.num_scheduled_tokens
            self.kv.sync_prefix(seq)
            seq.num_scheduled_tokens = 0

            if is_prefill and seq.num_cached_tokens < seq.num_prompt_tokens:
                outputs.append(EngineCoreOutput(seq_id=seq.seq_id))
                continue

            if seq.max_tokens <= 0:
                seq.status = SequenceStatus.FINISHED
                self.kv.deallocate(seq, publish=True)
                finished.add(seq.seq_id)
                outputs.append(EngineCoreOutput(seq_id=seq.seq_id, is_finished=True))
                continue

            seq.append_token(token_id)
            # 投机开启但本步走普通解码时，也要把新 token 喂给提案器
            if not is_prefill and self.spec_proposer is not None:
                self.spec_proposer.update(seq, [token_id])

            hit_eos = (not seq.ignore_eos) and token_id == self.eos
            hit_len = seq.num_completion_tokens >= seq.max_tokens
            if hit_eos or hit_len:
                seq.status = SequenceStatus.FINISHED
                self.kv.deallocate(seq, publish=True)
                finished.add(seq.seq_id)
                if self.spec_proposer is not None:
                    self.spec_proposer.drop(seq.seq_id)

            outputs.append(
                EngineCoreOutput(
                    seq_id=seq.seq_id,
                    new_token_ids=[token_id],
                    is_finished=seq.seq_id in finished,
                )
            )

        if finished:
            self.running = deque(s for s in self.running if s.seq_id not in finished)
        return outputs
