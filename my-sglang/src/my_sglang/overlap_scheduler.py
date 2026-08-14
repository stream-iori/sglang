"""CUDA-shaped overlap scheduler, executed deterministically on CPU."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import TextIO

import numpy as np

from my_sglang.models import ForwardBatch, ForwardMode, MemorySnapshot, Req, RequestStatus
from my_sglang.runner import FakeCudaRunner, FakeGenerationBatchResult
from my_sglang.schedule_batch import MiniScheduleBatch
from my_sglang.scheduler import MiniScheduler, StepResult


class FutureMap:
    """CPU 表示的 CUDA ``output_tokens_buf``，按稳定 request row 索引。"""

    def __init__(self, req_pool_size: int) -> None:
        self.output_tokens_buf = np.full(req_pool_size, -1, dtype=np.int64)
        self.valid = np.zeros(req_pool_size, dtype=bool)

    def stash(self, rows: np.ndarray, tokens: np.ndarray) -> None:
        self.output_tokens_buf[rows] = tokens
        self.valid[rows] = True

    def gather(self, rows: np.ndarray) -> np.ndarray:
        if not bool(self.valid[rows].all()):
            raise RuntimeError("FutureMap gather before producer token is published")
        tokens = self.output_tokens_buf[rows].copy()
        # Match SGLang's CI debug consume-once check: every gather needs a fresh
        # producer stash. Production SGLang omits this validity bookkeeping.
        self.valid[rows] = False
        return tokens

    def clear(self, row: int) -> None:
        self.output_tokens_buf[row] = -1
        self.valid[row] = False

    def snapshot(self) -> tuple[tuple[int, int], ...]:
        return tuple((int(row), int(self.output_tokens_buf[row])) for row in np.flatnonzero(self.valid))


@dataclass(frozen=True)
class PipelineJobState:
    job_id: int
    mode: ForwardMode
    rids: tuple[str, ...]


@dataclass(frozen=True)
class PipelineStepResult:
    launched_batches: tuple[ForwardBatch, ...] = ()
    processed_results: tuple[StepResult, ...] = ()
    queue: tuple[PipelineJobState, ...] = ()
    barrier_reason: str | None = None
    memory: MemorySnapshot | None = None

    @property
    def queue_depth(self) -> int:
        return len(self.queue)


@dataclass(frozen=True)
class _PipelineJob:
    job_id: int
    batch: MiniScheduleBatch
    forward: ForwardBatch
    result: FakeGenerationBatchResult
    retracted_rids: tuple[str, ...] = ()
    aborted_rids: tuple[str, ...] = ()

    def state(self) -> PipelineJobState:
        return PipelineJobState(self.job_id, self.batch.forward_mode, tuple(req.rid for req in self.batch.reqs))


class MiniOverlapScheduler(MiniScheduler):
    """对齐 SGLang 基础 CUDA overlap：launch current，再 FIFO process previous。"""

    def __init__(self, runner: FakeCudaRunner, *, max_running_reqs: int = 128,
                 max_total_tokens: int = 8192,
                 max_context_len: int | None = None, page_size: int = 1,
                 max_prefill_tokens: int | None = None, new_token_ratio: float = 0.5,
                 enable_radix_cache: bool = False, radix_cache=None,
                 chunked_prefill_size: int | None = None, trace: bool = True,
                 trace_file: TextIO | None = None):
        super().__init__(runner, max_running_reqs=max_running_reqs, max_total_tokens=max_total_tokens,
                         max_context_len=max_context_len, page_size=page_size,
                         max_prefill_tokens=max_prefill_tokens, new_token_ratio=new_token_ratio,
                         enable_radix_cache=enable_radix_cache, radix_cache=radix_cache,
                         chunked_prefill_size=chunked_prefill_size, trace=trace, trace_file=trace_file)
        self.runner: FakeCudaRunner = runner
        self.result_queue: deque[_PipelineJob] = deque()
        self.future_map = FutureMap(self.req_to_token_pool.req_to_token.shape[0])
        self._next_job_id = 0
        self._inflight_refs: Counter[Req] = Counter()
        self._deferred_finished: set[Req] = set()

    def result_queue_state(self) -> tuple[PipelineJobState, ...]:
        return tuple(job.state() for job in self.result_queue)

    def pipeline_step(self) -> PipelineStepResult:
        launched: list[ForwardBatch] = []
        processed: list[StepResult] = []
        barrier: str | None = None
        try:
            if not self.result_queue:
                job, administrative = self._schedule_and_launch_fresh()
                if job is not None:
                    launched.append(job.forward)
                if administrative is not None:
                    processed.append(administrative)
                return self._result(launched, processed, None)

            head = self.result_queue[0]
            successor: _PipelineJob | None = None
            if self._can_relay_decode(head):
                successor = self._launch_relay_decode(head)
                if successor is None:
                    barrier = "decode_memory"
                else:
                    self._enqueue(successor)
                    launched.append(successor.forward)
            else:
                barrier = self._barrier_reason(head)

            # This is the defining CUDA overlap order: B1 was submitted before B0
            # waits for its copy event and updates Python request state.
            processed.append(self._finalize_oldest())

            if successor is not None and any(
                req.status is RequestStatus.FINISHED for req in successor.batch.reqs
            ):
                barrier = "request_finished"
                processed.append(self._finalize_oldest())

            if not self.result_queue:
                job, administrative = self._schedule_and_launch_fresh()
                if job is not None:
                    launched.append(job.forward)
                if administrative is not None:
                    processed.append(administrative)
            self.assert_consistent()
            return self._result(launched, processed, barrier)
        except Exception:
            self._recover_pipeline_failure()
            raise

    def drain(self) -> tuple[StepResult, ...]:
        results: list[StepResult] = []
        try:
            while self.result_queue:
                results.append(self._finalize_oldest())
            self.assert_consistent()
            return tuple(results)
        except Exception:
            self._recover_pipeline_failure()
            raise

    def run_until_complete(self) -> list[Req]:
        completed: list[Req] = []
        known: dict[str, Req] = {}
        while self._has_work():
            for req in self._all_active_reqs():
                known[req.rid] = req
            turn = self.pipeline_step()
            for item in turn.processed_results:
                for rid in item.finished_rids + item.aborted_rids:
                    if rid in known and known[rid] not in completed:
                        completed.append(known[rid])
        return completed

    def _schedule_and_launch_fresh(self) -> tuple[_PipelineJob | None, StepResult | None]:
        self._settle_last_batch()
        retracted: list[str] = []
        aborted: list[str] = []
        batch = self._get_new_prefill_batch(aborted)
        if batch is None:
            batch = self._get_decode_batch(retracted, aborted)
        if batch is None:
            if not retracted and not aborted:
                return None, None
            administrative = StepResult(
                None, (), tuple(retracted), tuple(aborted), self.memory_snapshot()
            )
            return None, administrative
        job = self._launch(batch, tuple(retracted), tuple(aborted))
        self._enqueue(job)
        if batch.forward_mode is ForwardMode.DECODE:
            self.running_batch = self._empty_batch()
        return job, None

    def _launch_relay_decode(self, previous: _PipelineJob) -> _PipelineJob | None:
        reqs = [req for req in previous.batch.reqs if req.status is RequestStatus.RUNNING]
        if not reqs:
            return None
        needed = self._decode_pages_needed(reqs)
        self._evict_for_pages(needed)
        if needed > self._free_page_count():
            return None
        batch = MiniScheduleBatch(
            reqs,
            ForwardMode.DECODE,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
        )
        batch.prepare_for_decode()
        self._emit("future_gather_enqueue", rows=[req.req_pool_idx for req in reqs])
        return self._launch(batch, (), ())

    def _launch(self, batch: MiniScheduleBatch, retracted: tuple[str, ...], aborted: tuple[str, ...]) -> _PipelineJob:
        forward = batch.to_forward_batch()
        try:
            result = self.runner.run_batch_async(forward, self.future_map)
            batch.commit_allocated()
        except Exception:
            self._rollback_failed_batch(batch)
            raise
        job = _PipelineJob(
            self._next_job_id, batch, forward, result, retracted, aborted
        )
        self._next_job_id += 1
        self._emit(
            "pipeline_launch",
            job_id=job.job_id,
            mode=forward.forward_mode.value,
            rids=[req.rid for req in batch.reqs],
        )
        return job

    def _enqueue(self, job: _PipelineJob) -> None:
        if len(self.result_queue) >= 2:
            raise AssertionError("pipeline result queue depth exceeds two")
        self.result_queue.append(job)
        self._inflight_refs.update(job.batch.reqs)

    def _finalize_oldest(self) -> StepResult:
        job = self.result_queue[0]
        tokens = self.runner.resolve(job.result)
        finished: list[str] = []
        self._process_pipeline_result(job.batch, tokens, finished)
        self.result_queue.popleft()
        for req in job.batch.reqs:
            self._inflight_refs[req] -= 1
            if self._inflight_refs[req] <= 0:
                del self._inflight_refs[req]
                if req in self._deferred_finished:
                    self._release_deferred_finished(req)
        successors: set[Req] = set()
        if self.result_queue:
            successors = set(self.result_queue[0].batch.reqs)
        for req in job.batch.reqs:
            # 一个刚完成 EXTEND 的请求还未进入后继 decode batch 时，FutureMap
            # 中的首 token 必须保留；后继 launch 会读取它并用新 token 覆盖。
            if (
                req not in successors
                and req.req_pool_idx is not None
                and req.status is not RequestStatus.RUNNING
            ):
                self.future_map.clear(req.req_pool_idx)
            if req.status is RequestStatus.RUNNING and req not in successors:
                batch = MiniScheduleBatch(
                    [req],
                    ForwardMode.DECODE,
                    self.req_to_token_pool,
                    self.token_to_kv_pool_allocator,
                    self.tree_cache,
                )
                self.running_batch.merge_batch(batch)
        if job.batch.forward_mode is ForwardMode.DECODE:
            self.last_decode_batch = job.forward
        else:
            self.last_prefill_batch = job.forward
        self._emit("pipeline_process", job_id=job.job_id, finished=finished)
        return StepResult(
            job.forward,
            tuple(finished),
            job.retracted_rids,
            job.aborted_rids,
            self.memory_snapshot(),
        )

    def _process_pipeline_result(self, batch: MiniScheduleBatch, tokens: list[int], finished: list[str]) -> None:
        for req, token in zip(batch.reqs, tokens, strict=True):
            extend_range = self._require_extend_range(req)
            is_last = extend_range.end >= len(req.get_fill_ids())
            if req.status is RequestStatus.FINISHED:
                self._emit("pipeline_drop_token", rid=req.rid, token=token)
            elif batch.forward_mode is ForwardMode.EXTEND and not is_last:
                req.status = RequestStatus.PREFILLING
                self.chunked_req = req
                self._cache_unfinished_req(req)
            else:
                if self.chunked_req is req:
                    self.chunked_req = None
                req.append_output(token)
                if req.maybe_finish():
                    self._defer_finish(req, finished)
                elif batch.forward_mode is ForwardMode.EXTEND:
                    req.mark_running()

    def _defer_finish(self, req: Req, finished: list[str]) -> None:
        req.status = RequestStatus.FINISHED
        self._deferred_finished.add(req)
        finished.append(req.rid)

    def _release_deferred_finished(self, req: Req) -> None:
        if req.req_pool_idx is not None:
            self.future_map.clear(req.req_pool_idx)
        self._release_active_memory(req, keep_cache=False)
        self.runner.remove_request(req.rid)
        self._deferred_finished.discard(req)

    def _can_relay_decode(self, head: _PipelineJob) -> bool:
        return self._barrier_reason(head) is None

    def _barrier_reason(self, head: _PipelineJob) -> str | None:
        if head.batch.forward_mode is not ForwardMode.DECODE:
            return "non_decode"
        if self.waiting_queue or self.chunked_req is not None:
            return "prefill_waiting"
        if any(req.status is RequestStatus.FINISHED for req in head.batch.reqs):
            return "request_finished"
        return None

    def _recover_pipeline_failure(self) -> None:
        jobs = list(self.result_queue)
        affected = list(dict.fromkeys(req for job in jobs for req in job.batch.reqs))
        for job in reversed(jobs):
            try:
                job.result.discard()
            except Exception:
                pass
        self.result_queue.clear()
        self._inflight_refs.clear()
        self.running_batch = self._empty_batch()
        self.last_batch = None
        for req in affected:
            if req.req_pool_idx is not None:
                self.future_map.clear(req.req_pool_idx)
            self.runner.remove_request(req.rid)
            self._release_active_memory(req, keep_cache=True)
            self._deferred_finished.discard(req)
            if req.status is not RequestStatus.FINISHED:
                req.reset_for_retract()
                if req not in self.waiting_queue:
                    self.waiting_queue.append(req)
        if self.chunked_req in affected:
            self.chunked_req = None

    def _result(self, launched: list[ForwardBatch], processed: list[StepResult], barrier: str | None) -> PipelineStepResult:
        return PipelineStepResult(
            tuple(launched),
            tuple(processed),
            self.result_queue_state(),
            barrier,
            self.memory_snapshot(),
        )

    def _empty_batch(self) -> MiniScheduleBatch:
        return MiniScheduleBatch.empty(self.req_to_token_pool, self.token_to_kv_pool_allocator, self.tree_cache)

    def assert_consistent(self) -> None:
        super().assert_consistent()
        if len(self.result_queue) > 2:
            raise AssertionError("pipeline result queue depth exceeds two")
        expected = Counter(req for job in self.result_queue for req in job.batch.reqs)
        if expected != self._inflight_refs:
            raise AssertionError("pipeline in-flight request accounting mismatch")

    def _has_work(self) -> bool:
        return bool(super()._has_work() or self.result_queue or self._deferred_finished)

    def _all_active_reqs(self) -> list[Req]:
        reqs = super()._all_active_reqs()
        reqs.extend(req for job in self.result_queue for req in job.batch.reqs)
        reqs.extend(self._deferred_finished)
        return list(dict.fromkeys(reqs))
