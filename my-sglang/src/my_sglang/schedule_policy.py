from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from my_sglang.models import Req
from my_sglang.pools import BaseTokenToKVPoolAllocator
from my_sglang.schedule_batch import MiniScheduleBatch


class AddReqResult(str, Enum):
    ADMIT = "admit"
    CHUNK = "chunk"
    DEFER = "defer"
    ABORT = "abort"


@dataclass(frozen=True)
class AdmissionDecision:
    req: Req
    result: AddReqResult
    prefix_len: int
    target_fill_len: int
    reason: str | None = None


@dataclass
class MemoryBudget:
    free_tokens: int
    evictable_tokens: int
    protected_tokens: int
    decode_reserved_tokens: int
    remaining_prefill_tokens: int
    remaining_tokens: int


class PrefillAdder:
    """教学版 admission controller，对应 SGLang 的 PrefillAdder 主决策。"""

    def __init__(
        self,
        allocator: BaseTokenToKVPoolAllocator,
        tree_cache,
        running_batch: MiniScheduleBatch,
        *,
        max_prefill_tokens: int,
        chunked_prefill_size: int | None,
        new_token_ratio: float,
    ):
        self.allocator = allocator
        self.tree_cache = tree_cache
        self.running_batch = running_batch
        self.max_prefill_tokens = max_prefill_tokens
        self.chunked_prefill_size = chunked_prefill_size
        self.new_token_ratio = new_token_ratio
        self.has_running_reqs = not running_batch.is_empty()
        self.num_accepted = 0
        decode_reserved = sum(
            self._round_page(
                math.ceil(
                    req.remaining_new_tokens
                    * (1.0 if req.retracted_stain else self.new_token_ratio)
                )
            )
            for req in running_batch.reqs
        )
        evictable = tree_cache.evictable_size() if tree_cache is not None else 0
        protected = tree_cache.protected_size() if tree_cache is not None else 0
        available = allocator.available_size + evictable
        self.budget = MemoryBudget(
            free_tokens=allocator.available_size,
            evictable_tokens=evictable,
            protected_tokens=protected,
            decode_reserved_tokens=decode_reserved,
            remaining_prefill_tokens=max_prefill_tokens,
            remaining_tokens=max(available - decode_reserved, 0),
        )

    def add_requests(
        self,
        waiting_reqs: list[Req],
        chunked_req: Req | None,
        max_new_reqs: int,
    ) -> list[AdmissionDecision]:
        decisions: list[AdmissionDecision] = []
        candidates = ([chunked_req] if chunked_req is not None else []) + waiting_reqs
        accepted = 0
        for req in candidates:
            if req is None:
                continue
            if accepted >= max_new_reqs and req is not chunked_req:
                decisions.append(
                    AdmissionDecision(req, AddReqResult.DEFER, 0, 0, "request rows full")
                )
                continue
            decision = self._decide(req, continuing=req is chunked_req)
            decisions.append(decision)
            if decision.result in (AddReqResult.ADMIT, AddReqResult.CHUNK):
                accepted += 1
                self._consume(req, decision)
                if decision.result is AddReqResult.CHUNK:
                    # 教学版保持一个 chunked_req，避免同时维护多条中间状态链。
                    break
            elif decision.result is AddReqResult.DEFER:
                # FCFS：第一个因预算 defer 后，后续 waiting 不再越过它。
                if req is not chunked_req:
                    break
        return decisions

    def _decide(self, req: Req, *, continuing: bool) -> AdmissionDecision:
        # 决策顺序：先算 cache prefix，再尝试固定 chunk，然后尝试整个
        # suffix，最后在空系统/续 chunk 上使用物理可容纳性作防饿饿兜底。
        total_len = len(req.fill_ids)
        prefix_len = req.kv_committed_len if continuing else self._match_len(req)
        if prefix_len >= total_len:
            return AdmissionDecision(req, AddReqResult.ADMIT, prefix_len, total_len)

        full_extend = total_len - prefix_len
        if (
            self.chunked_prefill_size is not None
            and full_extend > self.chunked_prefill_size
        ):
            chunk_len = self.chunked_prefill_size
            if self.budget.remaining_prefill_tokens < chunk_len:
                chunk_len = self.budget.remaining_prefill_tokens
            if chunk_len > 0:
                target = prefix_len + chunk_len
                cost = self._candidate_cost(req, prefix_len, target, is_last=False)
                if cost <= self.budget.remaining_tokens:
                    return AdmissionDecision(
                        req, AddReqResult.CHUNK, prefix_len, target
                    )

        full_cost = self._candidate_cost(req, prefix_len, total_len, is_last=True)
        # 不能用 remaining_prefill_tokens 判断“首请求”：full cache hit 的 extend
        # 长度为 0，不会消耗该预算，却仍然已经占用了一个 request row/输出预留。
        first_admission = self.num_accepted == 0
        throughput_ok = (
            full_extend <= self.budget.remaining_prefill_tokens or first_admission
        )
        if full_cost <= self.budget.remaining_tokens and throughput_ok:
            return AdmissionDecision(req, AddReqResult.ADMIT, prefix_len, total_len)

        physical_tokens = self.allocator.available_size + (
            self.tree_cache.evictable_size() if self.tree_cache else 0
        )
        if first_admission and (continuing or not self.has_running_reqs):
            # 空系统的首请求，或已经开始的 chunk，不能因为“未来
            # decode 预留 + 对齐余量”而永久 defer。只要本次真实扩展能
            # 放进物理池就允许运行，后续压力由 evict/retract/abort 闭环处理。
            fallback_extend = (
                min(full_extend, self.chunked_prefill_size)
                if self.chunked_prefill_size is not None
                else full_extend
            )
            if self._round_page(fallback_extend) <= physical_tokens:
                target = prefix_len + fallback_extend
                result = (
                    AddReqResult.ADMIT
                    if target >= total_len
                    else AddReqResult.CHUNK
                )
                return AdmissionDecision(req, result, prefix_len, target)

        if self.chunked_prefill_size is not None:
            max_chunk = min(full_extend, self.chunked_prefill_size)
            if not first_admission:
                max_chunk = min(max_chunk, self.budget.remaining_prefill_tokens)
            for chunk_len in range(max_chunk, 0, -1):
                target = prefix_len + chunk_len
                cost = self._candidate_cost(
                    req, prefix_len, target, is_last=target >= total_len
                )
                if cost <= self.budget.remaining_tokens:
                    result = (
                        AddReqResult.ADMIT
                        if target >= total_len
                        else AddReqResult.CHUNK
                    )
                    return AdmissionDecision(req, result, prefix_len, target)

        minimum_cost = self._round_page(1)
        if first_admission and (
            minimum_cost > physical_tokens
            or (
                self.chunked_prefill_size is None
                and self._round_page(full_extend) > physical_tokens
            )
        ):
            return AdmissionDecision(
                req,
                AddReqResult.ABORT,
                prefix_len,
                prefix_len,
                "request cannot fit one KV page",
            )
        return AdmissionDecision(
            req, AddReqResult.DEFER, prefix_len, prefix_len, "prefill budget exhausted"
        )

    def _candidate_cost(
        self, req: Req, prefix_len: int, target: int, *, is_last: bool
    ) -> int:
        extend_cost = self._round_page(max(target - prefix_len, 0))
        reserve_ratio = 1.0 if req.retracted_stain else self.new_token_ratio
        output_reserve = (
            self._round_page(math.ceil(req.remaining_new_tokens * reserve_ratio))
            if is_last
            else 0
        )
        # 与生产 PrefillAdder 一样，为每个新请求保留一页对齐余量。
        return extend_cost + output_reserve + self.allocator.page_size

    def _consume(self, req: Req, decision: AdmissionDecision) -> None:
        is_last = decision.target_fill_len >= len(req.fill_ids)
        cost = self._candidate_cost(
            req, decision.prefix_len, decision.target_fill_len, is_last=is_last
        )
        extend = decision.target_fill_len - decision.prefix_len
        self.budget.remaining_tokens = max(self.budget.remaining_tokens - cost, 0)
        self.budget.remaining_prefill_tokens = max(
            self.budget.remaining_prefill_tokens - extend, 0
        )
        self.num_accepted += 1

    def _match_len(self, req: Req) -> int:
        if self.tree_cache is None:
            return 0
        return self.tree_cache.match_prefix(req.fill_ids, pin=False).token_count

    def _round_page(self, tokens: int) -> int:
        if tokens <= 0:
            return 0
        page = self.allocator.page_size
        return ((tokens + page - 1) // page) * page
