from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TextIO

from my_sglang.models import BatchForward, ForwardMode, MemorySnapshot, Req
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


class MiniOverlapScheduler(MiniScheduler):
    """显式 launch/finalize 的 MLX lazy scheduler。"""

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
        trace: bool = False,
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
        self._pending: PendingOverlapStep | None = None

    @property
    def pending(self) -> PendingOverlapStep | None:
        return self._pending

    def launch_step(self) -> OverlapLaunchResult:
        # 两阶段时序见 docs/scheduler-kv-overview.md#6-overlap-两阶段。
        # launch 允许 allocated > committed，但不允许在 pending 期间再调度一批。
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
                        self.runner.prefill_kick(handle)
                    else:
                        handle = self.runner.extend_start(
                            req_id=req.rid,
                            new_token_ids=new_ids,
                            new_slot_ids=new_slots,
                        )
                        self.runner.extend_kick(handle)
                    extends.append(PendingExtend(req, handle, first))
            else:
                handle = self.runner.decode_batch_start(
                    [req.rid for req in batch.reqs]
                )
                self.runner.decode_batch_kick(handle)
                pending_decode = PendingDecode(tuple(batch.reqs), handle)
        except Exception:
            self._rollback_failed_batch(batch)
            raise

        # 此处只完成 allocation/start/kick，committed 必须等 finalize。
        self._pending = PendingOverlapStep(
            batch=batch,
            forward=forward,
            extends=tuple(extends),
            decode=pending_decode,
            retracted_rids=tuple(retracted),
            aborted_rids=tuple(aborted),
        )
        self.assert_consistent()
        return OverlapLaunchResult(
            forward, tuple(retracted), tuple(aborted), self.memory_snapshot()
        )

    def finalize_pending(self) -> StepResult:
        # 只有 lazy token 成功取回后才 commit；任何异常都必须走与
        # normal scheduler 相同的 allocated-but-uncommitted rollback。
        if self._pending is None:
            raise RuntimeError("launch_step must be called before finalize_pending")
        pending = self._pending
        batch = pending.batch
        finished: list[str] = []
        try:
            if pending.decode is not None:
                tokens = self.runner.decode_batch_finalize(pending.decode.handle)
                if len(tokens) != len(pending.decode.reqs):
                    raise RuntimeError("runner returned wrong decode batch size")
                tokens = [int(token) for token in tokens]
            else:
                tokens = []
                for item in pending.extends:
                    token = (
                        self.runner.prefill_finalize(item.handle)
                        if item.first
                        else self.runner.extend_finalize(item.handle)
                    )
                    tokens.append(int(token))
            batch.commit_allocated()
        except Exception:
            self._pending = None
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

    def run_until_complete(self) -> list[Req]:
        completed: list[Req] = []
        known: dict[str, Req] = {}
        while self._has_work() or self._pending is not None:
            for req in self._all_active_reqs():
                known[req.rid] = req
            if self._pending is None:
                launch = self.launch_step()
                for rid in launch.aborted_rids:
                    req = known.get(rid)
                    if req is not None and req not in completed:
                        completed.append(req)
                if launch.batch is None:
                    continue
            result = self.finalize_pending()
            for rid in result.finished_rids + result.aborted_rids:
                req = known.get(rid)
                if req is not None and req not in completed:
                    completed.append(req)
        return completed

    def _all_active_reqs(self) -> list[Req]:
        reqs = super()._all_active_reqs()
        if self._pending is not None:
            reqs.extend(self._pending.batch.reqs)
        return list(dict.fromkeys(reqs))
