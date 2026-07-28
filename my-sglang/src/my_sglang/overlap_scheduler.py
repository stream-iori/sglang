"""教学版 overlap scheduler。

阅读主线时先记住两个 batch：B0 是队首、其结果尚未回到 CPU；B1 是后继
decode。稳定态的一次 ``pipeline_step`` 按以下顺序执行：

1. 用 B0 的设备侧 future token 启动 B1；
2. 再 finalize 并处理 B0；
3. 队列留下 B1，下一轮把它当成新的 B0。

关键不变量是“计算可以向前跑，CPU 可见状态必须按 FIFO 提交”。因此请求完成、
KV 释放和异常恢复都不能只看单个 batch，而要同时考虑 result_queue 中的所有者。
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Literal, TextIO

from my_sglang.models import (
    BatchForward,
    ForwardMode,
    FutureTokenRef,
    MemorySnapshot,
    Req,
    RequestStatus,
)
from my_sglang.runner import LazyRunnerProtocol
from my_sglang.schedule_batch import MiniScheduleBatch
from my_sglang.scheduler import MiniScheduler, StepResult


@dataclass(frozen=True)
class PendingExtend:
    req: Req
    handle: Any
    first: bool


@dataclass(frozen=True)
class PendingDecode:
    reqs: tuple[Req, ...]
    handle: Any


@dataclass(frozen=True)
class PendingOverlapStep:
    batch: MiniScheduleBatch
    forward: BatchForward
    extends: tuple[PendingExtend, ...] = ()
    decode: PendingDecode | None = None
    retracted_rids: tuple[str, ...] = ()
    aborted_rids: tuple[str, ...] = ()


@dataclass(frozen=True)
class OverlapLaunchResult:
    batch: BatchForward | None
    retracted_rids: tuple[str, ...]
    aborted_rids: tuple[str, ...]
    memory: MemorySnapshot


@dataclass(frozen=True)
class PipelineJobState:
    job_id: int
    kind: Literal["fresh", "chained"]
    mode: ForwardMode
    rids: tuple[str, ...]
    future_outputs: tuple[FutureTokenRef, ...]


@dataclass(frozen=True)
class PipelineStepResult:
    """一次生产形态 overlap turn 的可观察结果。"""

    launched_batches: tuple[BatchForward, ...] = ()
    processed_results: tuple[StepResult, ...] = ()
    queue: tuple[PipelineJobState, ...] = ()
    barrier_reason: str | None = None
    memory: MemorySnapshot | None = None

    @property
    def queue_depth(self) -> int:
        return len(self.queue)


class MiniFutureMap:
    """按 request row 保存下一轮可消费的设备侧 token 引用。

    这里保存的是依赖关系元数据，不是 CPU 上的 token 值。真实 token 仍留在
    runner 的 lazy handle/设备计算图中；后继 decode 通过引用与前驱计算相连。
    """

    def __init__(self) -> None:
        self._refs: dict[int, FutureTokenRef] = {}

    def publish(self, refs: tuple[FutureTokenRef, ...]) -> None:
        for ref in refs:
            self._refs[ref.req_pool_idx] = ref

    def get(self, req_pool_idx: int) -> FutureTokenRef:
        return self._refs[req_pool_idx]

    def clear(self, req_pool_idx: int) -> None:
        self._refs.pop(req_pool_idx, None)

    def snapshot(self) -> tuple[FutureTokenRef, ...]:
        return tuple(self._refs[index] for index in sorted(self._refs))


@dataclass(frozen=True)
class _PipelineJob:
    """result_queue 中的一个未处理结果，也是其请求资源的临时所有者。"""

    job_id: int
    kind: Literal["fresh", "chained"]
    batch: MiniScheduleBatch
    forward: BatchForward
    extends: tuple[PendingExtend, ...]
    decode: PendingDecode | None
    future_outputs: tuple[FutureTokenRef, ...]
    retracted_rids: tuple[str, ...] = ()
    aborted_rids: tuple[str, ...] = ()

    def state(self) -> PipelineJobState:
        return PipelineJobState(
            job_id=self.job_id,
            kind=self.kind,
            mode=self.batch.forward_mode,
            rids=tuple(req.rid for req in self.batch.reqs),
            future_outputs=self.future_outputs,
        )


class MiniOverlapScheduler(MiniScheduler):
    """同时提供事务显微镜与生产形态 result-queue 流水的教学调度器。

    ``launch_step/finalize_pending`` 把一次 forward 拆成两半，适合观察 KV 的
    allocated/committed 边界；``pipeline_step`` 则演示生产主线。两种 driver
    对状态提交时机的定义不同，所以同一个实例不能混用。
    """

    def __init__(
        self,
        runner: LazyRunnerProtocol,
        *,
        max_running_reqs: int = 128,
        max_total_tokens: int = 8192,
        max_context_len: int | None = None,
        page_size: int = 1,
        max_prefill_tokens: int | None = None,
        new_token_ratio: float = 0.5,
        enable_radix_cache: bool = False,
        radix_cache=None,
        chunked_prefill_size: int | None = None,
        trace: bool = True,
        trace_file: TextIO | None = None,
    ):
        super().__init__(
            runner,
            max_running_reqs=max_running_reqs,
            max_total_tokens=max_total_tokens,
            max_context_len=max_context_len,
            page_size=page_size,
            max_prefill_tokens=max_prefill_tokens,
            new_token_ratio=new_token_ratio,
            enable_radix_cache=enable_radix_cache,
            radix_cache=radix_cache,
            chunked_prefill_size=chunked_prefill_size,
            trace=trace,
            trace_file=trace_file,
        )
        self.runner: LazyRunnerProtocol = runner

        # Manual driver：保留原来的 launch -> finalize 事务教学边界。
        self._pending: PendingOverlapStep | None = None

        # Pipeline driver：队列在稳定态跨 turn 保留一个 job，并在 turn 内短暂
        # 达到两个 job，从而先 launch B1，再 process B0。
        self._result_queue: deque[_PipelineJob] = deque()
        self._future_map = MiniFutureMap()
        self._next_job_id = 0
        self._inflight_refs: Counter[Req] = Counter()
        self._deferred_finished: set[Req] = set()
        self._driver_mode: Literal["manual", "pipeline"] | None = None

    @property
    def pending(self) -> PendingOverlapStep | None:
        return self._pending

    @property
    def result_queue(self) -> tuple[PipelineJobState, ...]:
        return tuple(job.state() for job in self._result_queue)

    @property
    def future_map(self) -> tuple[FutureTokenRef, ...]:
        return self._future_map.snapshot()

    def _select_driver(self, mode: Literal["manual", "pipeline"]) -> None:
        if self._driver_mode is None:
            self._driver_mode = mode
            return
        if self._driver_mode != mode:
            raise RuntimeError(
                f"cannot mix {self._driver_mode} and {mode} overlap drivers "
                "on one scheduler"
            )

    # ------------------------------------------------------------------
    # Manual transaction microscope (backward-compatible API)
    # ------------------------------------------------------------------
    def launch_step(self) -> OverlapLaunchResult:
        self._select_driver("manual")
        if self._pending is not None:
            raise RuntimeError("finalize_pending must be called before next launch")
        self._settle_last_batch()
        retracted: list[str] = []
        aborted: list[str] = []
        batch = self._get_new_prefill_batch(aborted)
        if batch is None:
            batch = self._get_decode_batch(retracted, aborted)
        if batch is None:
            self.assert_consistent()
            return OverlapLaunchResult(
                None, tuple(retracted), tuple(aborted), self.memory_snapshot()
            )

        try:
            forward, extends, pending_decode = self._start_fresh_batch(batch)
        except Exception:
            self._rollback_failed_batch(batch)
            raise

        self._pending = PendingOverlapStep(
            batch=batch,
            forward=forward,
            extends=extends,
            decode=pending_decode,
            retracted_rids=tuple(retracted),
            aborted_rids=tuple(aborted),
        )
        self.assert_consistent()
        return OverlapLaunchResult(
            forward, tuple(retracted), tuple(aborted), self.memory_snapshot()
        )

    def finalize_pending(self) -> StepResult:
        self._select_driver("manual")
        if self._pending is None:
            raise RuntimeError("launch_step must be called before finalize_pending")
        pending = self._pending
        batch = pending.batch
        finished: list[str] = []
        try:
            tokens = self._finalize_handles(pending.extends, pending.decode)
            batch.commit_allocated()
        except Exception:
            self._pending = None
            self._rollback_failed_batch(batch)
            raise

        self._process_batch_result(batch, tokens, finished)
        self.last_batch = batch
        if batch.forward_mode is ForwardMode.DECODE:
            self.running_batch = self._empty_batch()
            self.last_decode_batch = pending.forward
        else:
            self.last_prefill_batch = pending.forward
        self._pending = None
        self.assert_consistent()
        return StepResult(
            pending.forward,
            tuple(finished),
            pending.retracted_rids,
            pending.aborted_rids,
            self.memory_snapshot(),
        )

    def step(self) -> StepResult:
        launch = self.launch_step()
        if launch.batch is None:
            return StepResult(
                None,
                (),
                launch.retracted_rids,
                launch.aborted_rids,
                launch.memory,
            )
        return self.finalize_pending()

    # ------------------------------------------------------------------
    # Production-shaped result queue pipeline
    # ------------------------------------------------------------------
    def pipeline_step(self) -> PipelineStepResult:
        """推进一次生产式流水；通常 launch B1 后才处理队首 B0。

        队列为空时只负责填入首个 job。队列非空时先尝试 chain，再严格按 FIFO
        处理队首。这样把 CPU 的采样结果处理与下一轮设备计算重叠起来。
        """
        self._select_driver("pipeline")
        launched: list[BatchForward] = []
        processed: list[StepResult] = []
        barrier_reason: str | None = None

        try:
            # 冷启动：尚无可供 chain 的前驱，只能调度并发射第一个 batch。
            if not self._result_queue:
                fresh, administrative = self._schedule_and_launch_fresh()
                if fresh is not None:
                    launched.append(fresh.forward)
                if administrative is not None:
                    processed.append(administrative)
                self.assert_consistent()
                return self._pipeline_result(launched, processed, None)

            head = self._result_queue[0]
            barrier_reason = self._chain_barrier_reason(head)
            successor: _PipelineJob | None = None
            # 稳定态的核心：B0 仍在队首时先发射 B1。此刻队列会短暂变成
            # [B0, B1]，这是 overlap 真正发生的位置。
            if barrier_reason is None:
                successor = self._launch_chained_decode(head)
                if successor is None:
                    barrier_reason = "decode_memory"
                else:
                    self._enqueue_pipeline_job(successor)
                    launched.append(successor.forward)

            # 即使 B1 已先启动，CPU 状态仍必须按 FIFO 先提交 B0。
            head_result = self._finalize_oldest_pipeline_job()
            processed.append(head_result)

            # B1 已经提交而 B0 结算后发现请求 finished：B1 的计算不能取消。
            # 立即按 FIFO drain B1，丢掉 finished 请求的多余 token，随后再恢复
            # fresh scheduling。
            if successor is not None and any(
                req.status is RequestStatus.FINISHED for req in successor.batch.reqs
            ):
                barrier_reason = "request_finished"
                processed.append(self._finalize_oldest_pipeline_job())

            if not self._result_queue:
                # barrier 或主动 drain 使队列见底后，重新走普通调度入口；
                # 这里会再次给予新来的 prefill 请求调度机会。
                fresh, administrative = self._schedule_and_launch_fresh()
                if fresh is not None:
                    launched.append(fresh.forward)
                if administrative is not None:
                    processed.append(administrative)

            self.assert_consistent()
            return self._pipeline_result(launched, processed, barrier_reason)
        except Exception:
            self._recover_pipeline_failure()
            raise

    def drain(self) -> tuple[StepResult, ...]:
        """不发射新 batch，按 FIFO 把已经在途的结果全部提交到 CPU 状态。"""
        self._select_driver("pipeline")
        results: list[StepResult] = []
        try:
            while self._result_queue:
                results.append(self._finalize_oldest_pipeline_job())
            self.assert_consistent()
            return tuple(results)
        except Exception:
            self._recover_pipeline_failure()
            raise

    def run_until_complete(self) -> list[Req]:
        self._select_driver("pipeline")
        completed: list[Req] = []
        known: dict[str, Req] = {}
        while self._has_work():
            for req in self._all_active_reqs():
                known[req.rid] = req
            turn = self.pipeline_step()
            for result in turn.processed_results:
                for rid in result.finished_rids + result.aborted_rids:
                    req = known.get(rid)
                    if req is not None and req not in completed:
                        completed.append(req)
        return completed

    def _pipeline_result(
        self,
        launched: list[BatchForward],
        processed: list[StepResult],
        barrier_reason: str | None,
    ) -> PipelineStepResult:
        return PipelineStepResult(
            launched_batches=tuple(launched),
            processed_results=tuple(processed),
            queue=self.result_queue,
            barrier_reason=barrier_reason,
            memory=self.memory_snapshot(),
        )

    def _schedule_and_launch_fresh(
        self,
    ) -> tuple[_PipelineJob | None, StepResult | None]:
        retracted: list[str] = []
        aborted: list[str] = []
        batch = self._get_new_prefill_batch(aborted)
        if batch is None:
            batch = self._get_decode_batch(retracted, aborted)
        if batch is None:
            administrative = (
                StepResult(
                    None,
                    (),
                    tuple(retracted),
                    tuple(aborted),
                    self.memory_snapshot(),
                )
                if retracted or aborted
                else None
            )
            return None, administrative

        job = self._launch_fresh_pipeline_job(
            batch, tuple(retracted), tuple(aborted)
        )
        self._enqueue_pipeline_job(job)
        if batch.forward_mode is ForwardMode.DECODE:
            self.running_batch = self._empty_batch()
        return job, None

    def _launch_fresh_pipeline_job(
        self,
        batch: MiniScheduleBatch,
        retracted_rids: tuple[str, ...],
        aborted_rids: tuple[str, ...],
    ) -> _PipelineJob:
        """从普通 scheduler 选出的 batch 启动一条新的流水链。"""
        try:
            forward, extends, pending_decode = self._start_fresh_batch(batch)
        except Exception:
            self._rollback_failed_batch(batch)
            raise

        # 普通生产 overlap 在 prepare/launch 后已推进 KV 逻辑水位；result
        # queue 延迟的是 CPU output/finish/cache 处理，而不是普通 KV commit。
        batch.commit_allocated()
        job_id = self._allocate_job_id()
        futures = self._make_future_outputs(batch, job_id)
        return _PipelineJob(
            job_id=job_id,
            kind="fresh",
            batch=batch,
            forward=forward,
            extends=extends,
            decode=pending_decode,
            future_outputs=futures,
            retracted_rids=retracted_rids,
            aborted_rids=aborted_rids,
        )

    def _launch_chained_decode(self, previous: _PipelineJob) -> _PipelineJob | None:
        """让后继 decode 直接依赖前驱的设备侧输出，而不等待 CPU 读回。"""
        assert previous.decode is not None
        reqs = list(previous.batch.reqs)
        # chain 也会新增一个 token 的 KV，因此必须先完成与普通 decode 相同的
        # 容量检查。容量不够返回 None，由调用者把它解释为 decode_memory barrier。
        needed_pages = self._decode_pages_needed(reqs)
        self._evict_for_pages(needed_pages)
        if needed_pages > self._free_page_count():
            return None

        # 此时 previous 尚未 finalize，不能从 req.output_ids 读取“上一 token”。
        # FutureTokenRef 只描述去前驱 handle 的哪个输出位置取值。
        future_inputs = tuple(
            self._future_map.get(self._require_req_pool_idx(req)) for req in reqs
        )
        self._emit(
            "future_consume",
            previous_job_id=previous.job_id,
            refs=[ref.req_pool_idx for ref in future_inputs],
        )
        batch = MiniScheduleBatch(
            reqs=reqs,
            forward_mode=ForwardMode.DECODE,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            tree_cache=self.tree_cache,
        )
        batch.prepare_for_decode(future_inputs)
        forward = batch.to_forward_batch()
        handle = None
        try:
            # chained start 必须直接连接 previous.handle；若这里改成普通 start，
            # 就会退化成先把 token 同步回 CPU，再发射下一轮的假 overlap。
            handle = self.runner.decode_batch_start_chained(previous.decode.handle)
            self.runner.decode_batch_kick(handle)
        except Exception:
            if handle is not None:
                self._discard_pending(handle)
            batch.rollback_uncommitted()
            raise
        batch.commit_allocated()
        job_id = self._allocate_job_id()
        futures = self._make_future_outputs(batch, job_id)
        self._emit(
            "pipeline_chain_launch",
            job_id=job_id,
            previous_job_id=previous.job_id,
            rids=[req.rid for req in reqs],
        )
        return _PipelineJob(
            job_id=job_id,
            kind="chained",
            batch=batch,
            forward=forward,
            extends=(),
            decode=PendingDecode(tuple(reqs), handle),
            future_outputs=futures,
        )

    def _enqueue_pipeline_job(self, job: _PipelineJob) -> None:
        if len(self._result_queue) >= 2:
            raise AssertionError("pipeline result queue depth exceeds two")
        self._result_queue.append(job)
        self._future_map.publish(job.future_outputs)
        # 一个请求可能同时属于 B0 和 B1。计数器表达“还有几个在途 job 持有
        # 它”，请求即使逻辑完成，也必须等计数归零才能释放 row/KV。
        self._inflight_refs.update(job.batch.reqs)
        self._emit(
            "pipeline_enqueue",
            job_id=job.job_id,
            kind=job.kind,
            mode=job.batch.forward_mode.value,
            rids=[req.rid for req in job.batch.reqs],
            queue_depth=len(self._result_queue),
        )

    def _finalize_oldest_pipeline_job(self) -> StepResult:
        """提交队首 job，并在最后一个在途所有者离开时释放请求资源。"""
        if not self._result_queue:
            raise RuntimeError("pipeline result queue is empty")
        job = self._result_queue[0]
        # finalize 是设备结果首次变成 CPU token 的边界。
        tokens = self._finalize_handles(job.extends, job.decode)
        finished: list[str] = []
        self._process_pipeline_batch_result(job.batch, tokens, finished)
        self._result_queue.popleft()

        # 先减少所有权计数，再决定是否真正释放一个已完成请求。
        for req in job.batch.reqs:
            self._inflight_refs[req] -= 1
            if self._inflight_refs[req] <= 0:
                del self._inflight_refs[req]
                if req in self._deferred_finished:
                    self._release_deferred_finished(req)

        # 未完成且没有后继 job 接管的请求，重新放回 running_batch，等待普通
        # scheduler 在后续 turn 继续调度。
        successor_reqs = (
            set(self._result_queue[0].batch.reqs) if self._result_queue else set()
        )
        for req in job.batch.reqs:
            if req not in successor_reqs and req.req_pool_idx is not None:
                self._future_map.clear(req.req_pool_idx)
            if req.status is RequestStatus.RUNNING and req not in successor_reqs:
                self.running_batch.merge_batch(
                    MiniScheduleBatch(
                        [req],
                        ForwardMode.DECODE,
                        self.req_to_token_pool,
                        self.token_to_kv_pool_allocator,
                        self.tree_cache,
                    )
                )

        if job.batch.forward_mode is ForwardMode.DECODE:
            self.last_decode_batch = job.forward
        else:
            self.last_prefill_batch = job.forward
        self._emit(
            "pipeline_process",
            job_id=job.job_id,
            finished=finished,
            queue_depth=len(self._result_queue),
        )
        return StepResult(
            job.forward,
            tuple(finished),
            job.retracted_rids,
            job.aborted_rids,
            self.memory_snapshot(),
        )

    def _process_pipeline_batch_result(
        self, batch: MiniScheduleBatch, tokens: list[int], finished: list[str]
    ) -> None:
        if batch.forward_mode is ForwardMode.DECODE:
            for req, token in zip(batch.reqs, tokens, strict=True):
                if req.status is RequestStatus.FINISHED:
                    # B1 可能在 B0 告知 EOS/长度结束之前已经启动。B1 无法取消，
                    # 但其多算出的 token 绝不能进入用户可见的 output_ids。
                    self._emit("pipeline_drop_token", rid=req.rid, token=int(token))
                    continue
                req.append_output(token)
                if req.maybe_finish():
                    self._defer_finish(req, finished)
            return

        for req, token, is_last in zip(
            batch.reqs, tokens, batch.is_last_prefill_chunk, strict=True
        ):
            if req.status is RequestStatus.FINISHED:
                continue
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
                self._defer_finish(req, finished)
            else:
                req.mark_running()

    def _defer_finish(self, req: Req, finished: list[str]) -> None:
        # FINISHED 是逻辑状态，不等于资源已释放。只要后继 job 仍引用该请求，
        # req_pool row、KV 和 runner state 都必须保留到 inflight_refs 归零。
        req.status = RequestStatus.FINISHED
        self._deferred_finished.add(req)
        finished.append(req.rid)
        self._emit(
            "pipeline_defer_release",
            rid=req.rid,
            inflight_refs=self._inflight_refs[req],
            reason=req.finish_reason,
        )

    def _release_deferred_finished(self, req: Req) -> None:
        req_pool_idx = req.req_pool_idx
        if req_pool_idx is not None:
            self._future_map.clear(req_pool_idx)
        self._release_active_memory(req, keep_cache=False)
        self.runner.remove_request(req.rid)
        self._deferred_finished.discard(req)
        self._emit("pipeline_release_finished", rid=req.rid)

    def _chain_barrier_reason(self, head: _PipelineJob) -> str | None:
        """解释为什么当前队首不能安全地产生 chained decode 后继。"""
        if head.batch.forward_mode is not ForwardMode.DECODE or head.decode is None:
            return "non_decode"
        if self.waiting_queue or self.chunked_req is not None:
            return "prefill_waiting"
        if any(req.status is RequestStatus.FINISHED for req in head.batch.reqs):
            return "request_finished"
        if not hasattr(self.runner, "decode_batch_start_chained"):
            return "runner_no_chained_decode"
        return None

    def _make_future_outputs(
        self, batch: MiniScheduleBatch, job_id: int
    ) -> tuple[FutureTokenRef, ...]:
        # output_index 把 batch 内行号稳定地带给后继；req_pool_idx 则标识该行
        # 属于哪个请求。两者都不是 token 数值本身。
        refs = tuple(
            FutureTokenRef(
                req_pool_idx=self._require_req_pool_idx(req),
                producer_job_id=job_id,
                output_index=index,
            )
            for index, req in enumerate(batch.reqs)
        )
        self._emit(
            "future_publish",
            job_id=job_id,
            refs=[
                {
                    "req_pool_idx": ref.req_pool_idx,
                    "output_index": ref.output_index,
                }
                for ref in refs
            ],
        )
        return refs

    def _allocate_job_id(self) -> int:
        job_id = self._next_job_id
        self._next_job_id += 1
        return job_id

    def _recover_pipeline_failure(self) -> None:
        """丢弃整条在途链，只保留已经由更早 FIFO job 确认的逻辑输出。"""
        jobs = list(self._result_queue)
        affected = list(
            dict.fromkeys(req for job in jobs for req in job.batch.reqs)
        )
        # 后继依赖前驱，因此逆序同步/丢弃 handle，先拆消费者再拆生产者。
        for job in reversed(jobs):
            for item in job.extends:
                self._discard_pending(item.handle)
            if job.decode is not None:
                self._discard_pending(job.decode.handle)
        self._result_queue.clear()
        self._inflight_refs.clear()

        self.running_batch = self._empty_batch()
        self.last_batch = None
        # 在途 KV 虽已预留/提交给设备执行，却没有对应的 CPU 结果确认；释放
        # 物理状态后把未完成请求退回 WAITING，让已有 output_ids 参与 re-prefill。
        for req in affected:
            req_pool_idx = req.req_pool_idx
            if req_pool_idx is not None:
                self._future_map.clear(req_pool_idx)
            try:
                self.runner.remove_request(req.rid)
            except Exception:
                pass
            self._release_active_memory(req, keep_cache=True)
            self._deferred_finished.discard(req)
            if req.status is not RequestStatus.FINISHED:
                req.reset_for_retract()
                if req not in self.waiting_queue:
                    self.waiting_queue.append(req)
        if self.chunked_req in affected:
            self.chunked_req = None

    def _discard_pending(self, handle: Any) -> None:
        discard = getattr(self.runner, "discard_pending", None)
        if discard is None:
            return
        try:
            discard(handle)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Shared runner helpers and invariants
    # ------------------------------------------------------------------
    def _start_fresh_batch(
        self, batch: MiniScheduleBatch
    ) -> tuple[
        BatchForward, tuple[PendingExtend, ...], PendingDecode | None
    ]:
        forward = batch.to_forward_batch()
        extends: list[PendingExtend] = []
        pending_decode: PendingDecode | None = None
        try:
            if batch.forward_mode is ForwardMode.EXTEND:
                for index, req in enumerate(batch.reqs):
                    new_ids = list(batch.input_ids_by_req[index])
                    new_slots = [int(x) for x in batch.out_cache_locs_by_req[index]]
                    first = batch.first_extend_by_req[index]
                    if first:
                        handle = self.runner.prefill_start(
                            req_id=req.rid,
                            new_token_ids=new_ids,
                            full_token_ids=list(req.fill_ids[: req.fill_len]),
                            prefix_slot_ids=[int(x) for x in req.prefix_indices],
                            new_slot_ids=new_slots,
                            req_pool_idx=self._require_req_pool_idx(req),
                        )
                        extends.append(PendingExtend(req, handle, first))
                        self.runner.prefill_kick(handle)
                    else:
                        handle = self.runner.extend_start(
                            req_id=req.rid,
                            new_token_ids=new_ids,
                            new_slot_ids=new_slots,
                        )
                        extends.append(PendingExtend(req, handle, first))
                        self.runner.extend_kick(handle)
            else:
                handle = self.runner.decode_batch_start(
                    [req.rid for req in batch.reqs]
                )
                pending_decode = PendingDecode(tuple(batch.reqs), handle)
                self.runner.decode_batch_kick(handle)
        except Exception:
            for item in extends:
                self._discard_pending(item.handle)
            if pending_decode is not None:
                self._discard_pending(pending_decode.handle)
            raise
        return forward, tuple(extends), pending_decode

    def _finalize_handles(
        self,
        extends: tuple[PendingExtend, ...],
        pending_decode: PendingDecode | None,
    ) -> list[int]:
        if pending_decode is not None:
            tokens = self.runner.decode_batch_finalize(pending_decode.handle)
            if len(tokens) != len(pending_decode.reqs):
                raise RuntimeError("runner returned wrong decode batch size")
            return [int(token) for token in tokens]
        return [
            int(
                self.runner.prefill_finalize(item.handle)
                if item.first
                else self.runner.extend_finalize(item.handle)
            )
            for item in extends
        ]

    def _empty_batch(self) -> MiniScheduleBatch:
        return MiniScheduleBatch.empty(
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
        )

    def assert_consistent(self) -> None:
        """集中检查教学流水的三个不变量：深度、FIFO、请求所有权。"""
        super().assert_consistent()
        if len(self._result_queue) > 2:
            raise AssertionError("pipeline result queue depth exceeds two")
        expected_refs: Counter[Req] = Counter(
            req for job in self._result_queue for req in job.batch.reqs
        )
        if expected_refs != self._inflight_refs:
            raise AssertionError("pipeline in-flight request accounting mismatch")
        job_ids = [job.job_id for job in self._result_queue]
        if job_ids != sorted(job_ids) or len(job_ids) != len(set(job_ids)):
            raise AssertionError("pipeline jobs must remain FIFO and unique")
        for req in self._deferred_finished:
            if req.status is not RequestStatus.FINISHED:
                raise AssertionError("deferred release request is not finished")
            if self._inflight_refs[req] <= 0:
                raise AssertionError("finished request deferred without in-flight owner")

    def _has_work(self) -> bool:
        return bool(
            super()._has_work()
            or self._pending is not None
            or self._result_queue
            or self._deferred_finished
        )

    def _all_active_reqs(self) -> list[Req]:
        reqs = super()._all_active_reqs()
        if self._pending is not None:
            reqs.extend(self._pending.batch.reqs)
        for job in self._result_queue:
            reqs.extend(job.batch.reqs)
        reqs.extend(self._deferred_finished)
        return list(dict.fromkeys(reqs))
