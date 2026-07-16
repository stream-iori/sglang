from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import TextIO

import numpy as np

from my_sglang.models import (
    BatchForward,
    ForwardMode,
    MemorySnapshot,
    Req,
    RequestStatus,
)
from my_sglang.pools import ReqToTokenPool, build_token_allocator
from my_sglang.radix_cache import MiniRadixCache
from my_sglang.runner import RunnerProtocol
from my_sglang.schedule_batch import MiniScheduleBatch
from my_sglang.schedule_policy import AddReqResult, AdmissionDecision, PrefillAdder


@dataclass(frozen=True)
class StepResult:
    batch: BatchForward | None
    finished_rids: tuple[str, ...] = ()
    retracted_rids: tuple[str, ...] = ()
    aborted_rids: tuple[str, ...] = ()
    memory: MemorySnapshot | None = None

    @property
    def prefill_batch(self) -> BatchForward | None:
        return self.batch if self.batch and self.batch.mode is ForwardMode.EXTEND else None

    @property
    def decode_batch(self) -> BatchForward | None:
        return self.batch if self.batch and self.batch.mode is ForwardMode.DECODE else None


class MiniScheduler:
    """SGLang-like 单进程调度器：prefill 优先，否则 decode。"""

    def __init__(
        self,
        runner: RunnerProtocol,
        *,
        max_running_reqs: int = 128,
        max_total_tokens: int = 8192,
        max_context_len: int | None = None,
        page_size: int = 1,
        max_prefill_tokens: int | None = None,
        new_token_ratio: float = 0.5,
        enable_radix_cache: bool = False,
        radix_cache: MiniRadixCache | None = None,
        chunked_prefill_size: int | None = None,
        trace: bool = False,
        trace_file: TextIO | None = None,
    ):
        if not 0 <= new_token_ratio <= 1:
            raise ValueError("new_token_ratio must be between 0 and 1")
        self.runner = runner
        self.waiting_queue: list[Req] = []
        self.req_to_token_pool = ReqToTokenPool(
            max_running_reqs, max_context_len or max_total_tokens
        )
        self.token_to_kv_pool_allocator = build_token_allocator(
            max_total_tokens, page_size
        )
        self.tree_cache = radix_cache
        if self.tree_cache is None and enable_radix_cache:
            self.tree_cache = MiniRadixCache(page_size=page_size)
        if self.tree_cache is not None and self.tree_cache.page_size != page_size:
            raise ValueError("radix cache page_size must match allocator page_size")

        self.running_batch = MiniScheduleBatch.empty(
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
        )
        self.last_batch: MiniScheduleBatch | None = None
        self.chunked_req: Req | None = None
        self.max_prefill_tokens = max_prefill_tokens or max_total_tokens
        self.chunked_prefill_size = (
            chunked_prefill_size
            if chunked_prefill_size is not None and chunked_prefill_size > 0
            else None
        )
        self.new_token_ratio = new_token_ratio
        self.trace = trace
        self.trace_file = trace_file or sys.stderr
        self.last_prefill_batch: BatchForward | None = None
        self.last_decode_batch: BatchForward | None = None
        self._last_budget_decode_reserve = 0

        # 旧名字只作为只读学习入口；唯一真实结构是上面的 pool/allocator。
        self.req_pool = self.req_to_token_pool
        self.req_to_token = self.req_to_token_pool
        self.kv_pool = self.token_to_kv_pool_allocator
        self.radix_cache = self.tree_cache

    @property
    def prefilling_reqs(self) -> list[Req]:
        return [self.chunked_req] if self.chunked_req is not None else []

    @property
    def running_reqs(self) -> list[Req]:
        return self.running_batch.reqs

    def add_request(self, req: Req) -> None:
        active = self._all_active_reqs()
        if any(existing.rid == req.rid for existing in active):
            raise ValueError(f"duplicate rid {req.rid!r}")
        self.waiting_queue.append(req)
        self._emit("enqueue", rid=req.rid, prompt_len=len(req.origin_input_ids))

    def step(self) -> StepResult:
        # 主流程图：docs/scheduler-kv-overview.md#1-调度主循环。
        # 和生产 Scheduler 一样，last_batch 先结算；一次 step 只选
        # EXTEND 或 DECODE 之一，便于看清 batch 生命周期。
        finished: list[str] = []
        retracted: list[str] = []
        aborted: list[str] = []
        self._settle_last_batch()

        batch = self._get_new_prefill_batch(aborted)
        if batch is None:
            batch = self._get_decode_batch(retracted, aborted)
        if batch is None:
            result = StepResult(
                None,
                tuple(finished),
                tuple(retracted),
                tuple(aborted),
                self.memory_snapshot(),
            )
            self.assert_consistent()
            return result

        forward = batch.to_forward_batch()
        try:
            tokens = self._run_batch_sync(batch)
            batch.commit_allocated()
        except Exception:
            self._rollback_failed_batch(batch)
            raise

        self._process_batch_result(batch, tokens, finished)
        self.last_batch = batch
        if batch.forward_mode is ForwardMode.DECODE:
            self.running_batch = MiniScheduleBatch.empty(
                self.req_to_token_pool,
                self.token_to_kv_pool_allocator,
                self.tree_cache,
            )

        if batch.forward_mode is ForwardMode.EXTEND:
            self.last_prefill_batch = forward
        else:
            self.last_decode_batch = forward
        self.assert_consistent()
        return StepResult(
            forward,
            tuple(finished),
            tuple(retracted),
            tuple(aborted),
            self.memory_snapshot(),
        )

    def run_until_complete(self) -> list[Req]:
        completed: list[Req] = []
        known: dict[str, Req] = {}
        while self._has_work():
            for req in self._all_active_reqs():
                known[req.rid] = req
            result = self.step()
            for rid in result.finished_rids + result.aborted_rids:
                req = known.get(rid)
                if req is not None and req not in completed:
                    completed.append(req)
        return completed

    def _settle_last_batch(self) -> None:
        # last_batch 是“上一轮已 forward、本轮才归并”的 batch。
        # EXTEND 去掉 finished/chunked 后合入 running；DECODE 则直接过滤。
        if self.last_batch is None:
            return
        batch = self.last_batch
        if batch.forward_mode is ForwardMode.EXTEND:
            exclude = {self.chunked_req} if self.chunked_req is not None else set()
            batch.filter_batch(exclude=exclude)
            if not batch.is_empty():
                self.running_batch.merge_batch(batch)
        else:
            batch.filter_batch()
            self.running_batch = batch
        self.last_batch = None

    def _get_new_prefill_batch(
        self, aborted: list[str]
    ) -> MiniScheduleBatch | None:
        # 方法级数据流：PrefillAdder 只做决策，_attach_new_request
        # 绑定 row/cache prefix，MiniScheduleBatch.prepare_for_extend 才分配 slot。
        if not self.waiting_queue and self.chunked_req is None:
            return None

        max_rows = self.req_to_token_pool.available_size + (
            1 if self.chunked_req is not None else 0
        )
        adder = PrefillAdder(
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.running_batch,
            max_prefill_tokens=self.max_prefill_tokens,
            chunked_prefill_size=self.chunked_prefill_size,
            new_token_ratio=self.new_token_ratio,
        )
        self._last_budget_decode_reserve = adder.budget.decode_reserved_tokens
        decisions = adder.add_requests(
            self.waiting_queue, self.chunked_req, max_new_reqs=max_rows
        )
        accepted = [
            decision
            for decision in decisions
            if decision.result in (AddReqResult.ADMIT, AddReqResult.CHUNK)
        ]
        for decision in decisions:
            if decision.result is AddReqResult.ABORT:
                self._abort_waiting(decision.req, decision.reason or "admission failed")
                aborted.append(decision.req.rid)

        if not accepted:
            return None

        newly_attached: list[Req] = []
        try:
            for decision in accepted:
                req = decision.req
                continuing = req is self.chunked_req
                if not continuing:
                    self._attach_new_request(req)
                    newly_attached.append(req)
                req.fill_len = decision.target_fill_len
                req.extend_input_len = req.fill_len - req.kv_allocated_len

            needed = self._extend_pages_needed([d.req for d in accepted])
            self._evict_for_pages(needed)
            batch = MiniScheduleBatch(
                reqs=[decision.req for decision in accepted],
                forward_mode=ForwardMode.EXTEND,
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                tree_cache=self.tree_cache,
                chunked_req=self.chunked_req,
                first_extend_by_req=tuple(
                    decision.req in newly_attached for decision in accepted
                ),
            )
            batch.prepare_for_extend()
        except Exception:
            for req in reversed(newly_attached):
                self._release_active_memory(req, keep_cache=True)
                req.reset_for_retract()
            raise

        accepted_set = {decision.req for decision in accepted}
        self.waiting_queue = [req for req in self.waiting_queue if req not in accepted_set]
        self._emit(
            "prefill_admit",
            rids=[req.rid for req in batch.reqs],
            decisions=[decision.result.value for decision in accepted],
            remaining_tokens=adder.budget.remaining_tokens,
        )
        return batch

    def _get_decode_batch(
        self, retracted: list[str], aborted: list[str]
    ) -> MiniScheduleBatch | None:
        # decode 压力闭环：先 cache evict，再 retract 部分请求；
        # 若只剩一个请求仍无法获得一页，则明确 abort，不留半分配状态。
        reqs = [
            req for req in self.running_batch.reqs if req.status is RequestStatus.RUNNING
        ]
        if not reqs:
            return None

        while reqs:
            needed_pages = self._decode_pages_needed(reqs)
            self._evict_for_pages(needed_pages)
            if needed_pages <= self._free_page_count():
                break
            if len(reqs) == 1:
                req = reqs.pop()
                self._abort_active(req, "KV OOM after cache eviction and retraction")
                aborted.append(req.rid)
                break
            victim = min(
                reqs,
                key=lambda req: (req.generated_count, -len(req.full_token_ids)),
            )
            reqs.remove(victim)
            self._retract_req(victim)
            retracted.append(victim.rid)
            self._emit("retract", rid=victim.rid)

        if not reqs:
            self.running_batch.reqs = []
            return None

        self.running_batch.reqs = reqs
        self.running_batch.forward_mode = ForwardMode.DECODE
        self.running_batch.prepare_for_decode()
        return self.running_batch

    def _run_batch_sync(self, batch: MiniScheduleBatch) -> list[int]:
        if batch.forward_mode is ForwardMode.DECODE:
            tokens = self.runner.decode_batch([req.rid for req in batch.reqs])
            if len(tokens) != len(batch.reqs):
                raise RuntimeError("runner returned wrong decode batch size")
            return [int(token) for token in tokens]

        tokens: list[int] = []
        for index, req in enumerate(batch.reqs):
            new_ids = list(batch.input_ids_by_req[index])
            new_slots = [int(x) for x in batch.out_cache_locs_by_req[index]]
            if batch.first_extend_by_req[index]:
                token = self.runner.prefill(
                    req_id=req.rid,
                    new_token_ids=new_ids,
                    full_token_ids=list(req.fill_ids[: req.fill_len]),
                    prefix_slot_ids=[int(x) for x in req.prefix_indices],
                    new_slot_ids=new_slots,
                    req_pool_idx=self._require_req_pool_idx(req),
                )
            else:
                token = self.runner.extend(
                    req_id=req.rid,
                    new_token_ids=new_ids,
                    new_slot_ids=new_slots,
                )
            tokens.append(int(token))
        return tokens

    def _process_batch_result(
        self, batch: MiniScheduleBatch, tokens: list[int], finished: list[str]
    ) -> None:
        if batch.forward_mode is ForwardMode.DECODE:
            for req, token in zip(batch.reqs, tokens, strict=True):
                req.append_output(token)
                if req.maybe_finish():
                    self._finish_req(req, finished)
            return

        for req, token, is_last in zip(
            batch.reqs, tokens, batch.is_last_prefill_chunk, strict=True
        ):
            if not is_last:
                req.status = RequestStatus.PREFILLING
                self.chunked_req = req
                self._cache_unfinished_req(req)
                self._emit("prefill_chunk_done", rid=req.rid, fill_len=req.fill_len)
                continue

            if self.chunked_req is req:
                self.chunked_req = None
            req.append_output(token)
            if req.maybe_finish():
                self._finish_req(req, finished)
            else:
                req.mark_running()

    def _attach_new_request(self, req: Req) -> None:
        match = (
            self.tree_cache.match_prefix(req.fill_ids, pin=True)
            if self.tree_cache is not None
            else None
        )
        try:
            req.req_pool_idx = self.req_to_token_pool.alloc_one(req)
        except Exception:
            if match is not None and match.token_count:
                self.tree_cache.dec_lock_ref(match.last_node)
            raise
        if match is not None:
            req.prefix_indices = np.asarray(match.slot_ids, dtype=np.int64)
            req.last_node = match.last_node if match.token_count else None
            req.cache_protected_len = match.token_count
        prefix_len = len(req.prefix_indices)
        if prefix_len:
            self.req_to_token_pool.write(req.req_pool_idx, 0, req.prefix_indices)
        req.kv_allocated_len = prefix_len
        req.kv_committed_len = prefix_len

    def _cache_unfinished_req(self, req: Req) -> None:
        # 只把已 committed 的完整 page 交给 radix cache；未满页的尾部
        # 继续归请求所有。insert 可能返回已有 prefix 的 slot，因此要
        # 重写 req_to_token，并只释放不与新 prefix 共页的重复 page。
        if self.tree_cache is None or req.req_pool_idx is None:
            return
        cacheable_len = req.kv_committed_len // self.tree_cache.page_size * self.tree_cache.page_size
        if cacheable_len <= req.cache_protected_len:
            return
        old_slots = self.req_to_token_pool.row(req.req_pool_idx, cacheable_len)
        result = self.tree_cache.insert(req.fill_ids[:cacheable_len], old_slots)
        self.token_to_kv_pool_allocator.free(result.evicted_slots)
        if req.last_node is not None:
            self.tree_cache.dec_lock_ref(req.last_node)
        match = self.tree_cache.match_prefix(req.fill_ids[:cacheable_len], pin=True)
        new_slots = np.asarray(match.slot_ids, dtype=np.int64)
        self.token_to_kv_pool_allocator.free_unshared_pages(old_slots, new_slots)
        self.req_to_token_pool.write(req.req_pool_idx, 0, new_slots)
        req.prefix_indices = new_slots
        req.last_node = match.last_node
        req.cache_protected_len = len(new_slots)

    def _finish_req(self, req: Req, finished: list[str]) -> None:
        req.status = RequestStatus.FINISHED
        self._release_active_memory(req, keep_cache=False)
        self.runner.remove_request(req.rid)
        finished.append(req.rid)
        self._emit("finish", rid=req.rid, reason=req.finish_reason)

    def _release_active_memory(self, req: Req, *, keep_cache: bool) -> None:
        if req.req_pool_idx is None:
            return
        all_slots = self.req_to_token_pool.row(req.req_pool_idx, req.kv_allocated_len)
        protected_slots = np.empty((0,), dtype=np.int64)
        if self.tree_cache is not None:
            if req.last_node is not None:
                self.tree_cache.dec_lock_ref(req.last_node)
                req.last_node = None
            if not keep_cache:
                cacheable_len = (
                    req.kv_committed_len // self.tree_cache.page_size
                    * self.tree_cache.page_size
                )
                result = self.tree_cache.insert(
                    req.full_token_ids[:cacheable_len], all_slots[:cacheable_len]
                )
                self.token_to_kv_pool_allocator.free(result.evicted_slots)
                match = self.tree_cache.match_prefix(
                    req.full_token_ids[:cacheable_len], pin=False
                )
                protected_slots = np.asarray(match.slot_ids, dtype=np.int64)
            else:
                protected_slots = req.prefix_indices
        self.token_to_kv_pool_allocator.free_unshared_pages(
            all_slots, protected_slots
        )
        self.req_to_token_pool.free(req)
        req.req_pool_idx = None
        req.prefix_indices = np.empty((0,), dtype=np.int64)
        req.cache_protected_len = 0
        req.kv_allocated_len = 0
        req.kv_committed_len = 0
        req.fill_len = 0
        req.extend_input_len = 0

    def _retract_req(self, req: Req) -> None:
        # retract 保留 output_ids（逻辑进度），但丢弃 row/runner/非缓存 KV。
        # retracted_stain 会让下次 admission 按全部剩余输出做保守预留。
        self._release_active_memory(req, keep_cache=True)
        self.runner.remove_request(req.rid)
        req.reset_for_retract()
        self.waiting_queue.append(req)

    def _abort_active(self, req: Req, message: str) -> None:
        self._release_active_memory(req, keep_cache=True)
        self.runner.remove_request(req.rid)
        req.mark_aborted(message)

    def _abort_waiting(self, req: Req, message: str) -> None:
        if req in self.waiting_queue:
            self.waiting_queue.remove(req)
        req.mark_aborted(message)
        self._emit("abort", rid=req.rid, reason=message)

    def _rollback_failed_batch(self, batch: MiniScheduleBatch) -> None:
        # 事务边界：先 rollback allocated-but-uncommitted，再解绑请求。
        # normal runner 异常与 overlap finalize 异常共用这条恢复路径。
        batch.rollback_uncommitted()
        for req in batch.reqs:
            self.runner.remove_request(req.rid)
            self._release_active_memory(req, keep_cache=True)
            req.reset_for_retract()
            if req not in self.waiting_queue:
                self.waiting_queue.append(req)
        if self.chunked_req in batch.reqs:
            self.chunked_req = None

    def _extend_pages_needed(self, reqs: list[Req]) -> int:
        total = 0
        for req in reqs:
            last = self._last_loc(req)
            total += self.token_to_kv_pool_allocator.required_pages_for_extend(
                req.kv_allocated_len, req.fill_len, last
            )
        return total

    def _decode_pages_needed(self, reqs: list[Req]) -> int:
        return self.token_to_kv_pool_allocator.required_pages_for_decode(
            [req.kv_allocated_len + 1 for req in reqs],
            [self._last_loc(req) for req in reqs],
        )

    def _evict_for_pages(self, needed_pages: int) -> None:
        missing_pages = max(needed_pages - self._free_page_count(), 0)
        if missing_pages == 0 or self.tree_cache is None:
            return
        evicted = self.tree_cache.evict(
            missing_pages * self.token_to_kv_pool_allocator.page_size
        )
        self.token_to_kv_pool_allocator.free(evicted)
        if evicted:
            self._emit("cache_evict", slots=list(evicted))

    def _free_page_count(self) -> int:
        return (
            self.token_to_kv_pool_allocator.available_size
            // self.token_to_kv_pool_allocator.page_size
        )

    def _last_loc(self, req: Req) -> int:
        if req.req_pool_idx is None or req.kv_allocated_len == 0:
            return -1
        return int(
            self.req_to_token_pool.req_to_token[
                req.req_pool_idx, req.kv_allocated_len - 1
            ]
        )

    def memory_snapshot(self) -> MemorySnapshot:
        return MemorySnapshot(
            free_tokens=self.token_to_kv_pool_allocator.available_size,
            allocated_tokens=self.token_to_kv_pool_allocator.allocated_size,
            mapped_tokens=self.req_to_token_pool.mapped_size,
            cache_evictable_tokens=(
                self.tree_cache.evictable_size() if self.tree_cache else 0
            ),
            cache_protected_tokens=(
                self.tree_cache.protected_size() if self.tree_cache else 0
            ),
            decode_reserved_tokens=self._last_budget_decode_reserve,
        )

    def assert_consistent(self) -> None:
        self.req_to_token_pool.assert_consistent()
        self.token_to_kv_pool_allocator.assert_consistent()
        seen: set[Req] = set()
        for req in self._all_active_reqs():
            if req in seen or req.req_pool_idx is None:
                continue
            seen.add(req)
            if not 0 <= req.kv_committed_len <= req.kv_allocated_len:
                raise AssertionError("invalid committed/allocated KV boundary")
            slots = self.req_to_token_pool.row(req.req_pool_idx, req.kv_allocated_len)
            if np.any(slots < 0):
                raise AssertionError(f"request {req.rid} has unmapped KV positions")
            if any(
                not self.token_to_kv_pool_allocator.owns_slot(int(slot))
                for slot in slots
            ):
                raise AssertionError(f"request {req.rid} maps to a free KV page")
            if req.cache_protected_len > req.kv_committed_len:
                raise AssertionError("cache protected range exceeds committed KV")

    def _has_work(self) -> bool:
        return bool(
            self.waiting_queue
            or self.chunked_req is not None
            or not self.running_batch.is_empty()
            or self.last_batch is not None
        )

    def _all_active_reqs(self) -> list[Req]:
        reqs = [*self.waiting_queue, *self.running_batch.reqs]
        if self.last_batch is not None:
            reqs.extend(self.last_batch.reqs)
        if self.chunked_req is not None:
            reqs.append(self.chunked_req)
        # dict 保序去重；Req 采用 identity hash。
        return list(dict.fromkeys(reqs))

    @staticmethod
    def _require_req_pool_idx(req: Req) -> int:
        if req.req_pool_idx is None:
            raise RuntimeError(f"request {req.rid} has no req_pool_idx")
        return req.req_pool_idx

    def _emit(self, event: str, **payload) -> None:
        if not self.trace:
            return
        print(
            json.dumps({"event": event, **payload}, sort_keys=True),
            file=self.trace_file,
        )
