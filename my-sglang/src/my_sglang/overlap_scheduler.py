from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TextIO

from my_sglang.models import BatchForward, Req
from my_sglang.runner import LazyRunnerProtocol
from my_sglang.scheduler import MiniScheduler, StepResult


@dataclass(frozen=True)
class OverlapPrefillPlan:
    # overlap prefill 暂不接入 radix cache，因此一个 prompt token 对应一个新 KV slot。
    req: Req
    req_pool_idx: int
    slot_ids: tuple[int, ...]


@dataclass(frozen=True)
class PendingPrefill:
    # 一个请求的 prefill lazy handle，以及 prefill 前已经分配好的资源。
    req: Req
    req_pool_idx: int
    slot_ids: tuple[int, ...]
    handle: Any


@dataclass(frozen=True)
class PendingDecode:
    # 一个 decode batch 的 lazy handle。finalize 时会拿到每个请求的 next token。
    reqs: tuple[Req, ...]
    slots: tuple[int, ...]
    handle: Any


@dataclass(frozen=True)
class PendingOverlapStep:
    # overlap step 会先 launch，再 finalize；这个对象记录已 launch 但未 finalize 的工作。
    prefill_batch: BatchForward | None
    decode_batch: BatchForward | None
    prefills: tuple[PendingPrefill, ...]
    decode: PendingDecode | None


class MiniOverlapScheduler(MiniScheduler):
    # MiniOverlapScheduler 演示 MLX overlap scheduling 的核心思想：
    # 1. start: 构建 lazy graph，尽早交给 MLX 后端排队。
    # 2. kick: 触发 async_eval，让 GPU/Metal 有机会开始跑。
    # 3. finalize: CPU 需要 token 时再阻塞读取结果，并更新 Req 生命周期。
    #
    # 这个教学版不做 SGLang 里更复杂的 chained decode / speculative launch，
    # 先保留“本轮刚 prefill 的请求下一轮再 decode”的正常语义。
    def __init__(
        self,
        runner: LazyRunnerProtocol,
        *,
        max_running_reqs: int = 128,
        max_total_tokens: int = 8192,
        trace: bool = False,
        trace_file: TextIO | None = None,
    ):
        super().__init__(
            runner,
            max_running_reqs=max_running_reqs,
            max_total_tokens=max_total_tokens,
            trace=trace,
            trace_file=trace_file,
        )
        self.runner: LazyRunnerProtocol = runner

    def step(self) -> StepResult:
        # 和 normal scheduler 一样，decode 只处理 step 开始时已经 running 的请求。
        decode_candidates = list(self.running_reqs)
        finished: list[str] = []

        # overlap 的关键：先 launch prefill/decode，把 lazy work 交给 runner；
        # 然后再 finalize。真实 MLX 下，kick 后 GPU 可以和 CPU 后续工作重叠。
        pending = self._launch_overlap_step(decode_candidates)
        self._finalize_overlap_step(pending, finished)

        if pending.prefill_batch is not None:
            self.last_prefill_batch = pending.prefill_batch
        if pending.decode_batch is not None:
            self.last_decode_batch = pending.decode_batch
        return StepResult(pending.prefill_batch, pending.decode_batch, tuple(finished))

    def _launch_overlap_step(self, decode_candidates: list[Req]) -> PendingOverlapStep:
        pending_prefills, prefill_batch = self._launch_prefill_waiting()
        pending_decode, decode_batch = self._launch_decode(decode_candidates)
        return PendingOverlapStep(
            prefill_batch=prefill_batch,
            decode_batch=decode_batch,
            prefills=tuple(pending_prefills),
            decode=pending_decode,
        )

    def _launch_prefill_waiting(self) -> tuple[list[PendingPrefill], BatchForward | None]:
        if not self.waiting_queue:
            return [], None

        reqs = self.waiting_queue
        self.waiting_queue = []
        planned: list[OverlapPrefillPlan] = []
        pending_prefills: list[PendingPrefill] = []

        try:
            for req in reqs:
                req_pool_idx = self.req_pool.alloc(req.rid)
                try:
                    slot_ids = tuple(self.kv_pool.alloc_many(len(req.origin_input_ids)))
                except Exception:
                    self.req_pool.free(req.rid)
                    raise
                planned.append(
                    OverlapPrefillPlan(
                        req=req,
                        req_pool_idx=req_pool_idx,
                        slot_ids=slot_ids,
                    )
                )

            batch = self._build_overlap_prefill_batch(planned)
            self._emit(
                "overlap_prefill_launch",
                rids=[req.rid for req in batch.reqs],
                seq_lens=list(batch.seq_lens),
            )

            for plan in planned:
                req = plan.req
                self._attach_overlap_prefill_plan(plan)

                handle = self.runner.prefill_start(
                    req_id=req.rid,
                    new_token_ids=list(req.origin_input_ids),
                    full_token_ids=list(req.origin_input_ids),
                    prefix_slot_ids=[],
                    new_slot_ids=list(plan.slot_ids),
                    req_pool_idx=plan.req_pool_idx,
                )
                self.runner.prefill_kick(handle)
                pending_prefills.append(
                    PendingPrefill(
                        req=req,
                        req_pool_idx=plan.req_pool_idx,
                        slot_ids=plan.slot_ids,
                        handle=handle,
                    )
                )
            return pending_prefills, batch
        except Exception:
            for pending in reversed(pending_prefills):
                self.runner.remove_request(pending.req.rid)
            for plan in reversed(planned):
                req = plan.req
                self.kv_pool.free_many(list(plan.slot_ids))
                self.req_to_token.remove_req(plan.req_pool_idx)
                self.req_pool.free(req.rid)
                req.req_pool_idx = None
                req.kv_slots.clear()
                req.prefix_slot_ids.clear()
                req.owned_kv_slots.clear()
            self.waiting_queue = reqs + self.waiting_queue
            raise

    def _build_overlap_prefill_batch(
        self,
        planned: list[OverlapPrefillPlan],
    ) -> BatchForward:
        return BatchForward(
            mode="prefill",
            reqs=tuple(plan.req for plan in planned),
            input_ids_by_req=tuple(
                tuple(plan.req.origin_input_ids) for plan in planned
            ),
            req_pool_indices=tuple(plan.req_pool_idx for plan in planned),
            out_cache_locs=tuple(plan.slot_ids for plan in planned),
            seq_lens=tuple(len(plan.req.origin_input_ids) for plan in planned),
        )

    def _attach_overlap_prefill_plan(self, plan: OverlapPrefillPlan) -> None:
        req = plan.req
        # launch 前先写入资源归属；finalize 时如果请求结束，可以复用基类的释放逻辑。
        req.req_pool_idx = plan.req_pool_idx
        req.kv_slots.extend(plan.slot_ids)
        req.owned_kv_slots.extend(plan.slot_ids)
        for offset, slot in enumerate(plan.slot_ids):
            self.req_to_token.set(plan.req_pool_idx, offset, slot)

    def _launch_decode(
        self,
        candidates: list[Req],
    ) -> tuple[PendingDecode | None, BatchForward | None]:
        decode_reqs = self._collect_decode_reqs(candidates)
        if not decode_reqs:
            return None, None

        slots = self.kv_pool.alloc_many(len(decode_reqs))
        allocated = list(zip(decode_reqs, slots, strict=True))
        batch = self._build_decode_batch(decode_reqs, allocated)
        self._emit(
            "overlap_decode_launch",
            rids=[req.rid for req in batch.reqs],
            input_ids=[ids[0] for ids in batch.input_ids_by_req],
        )

        for req, slot in zip(decode_reqs, slots, strict=True):
            req_pool_idx = self._require_req_pool_idx(req)
            req.kv_slots.append(slot)
            req.owned_kv_slots.append(slot)
            self.req_to_token.set(req_pool_idx, len(req.full_token_ids) - 1, slot)

        handle = self.runner.decode_batch_start([req.rid for req in decode_reqs])
        self.runner.decode_batch_kick(handle)
        return (
            PendingDecode(reqs=tuple(decode_reqs), slots=tuple(slots), handle=handle),
            batch,
        )

    def _finalize_overlap_step(
        self,
        pending: PendingOverlapStep,
        finished: list[str],
    ) -> None:
        # 先 finalize decode，再 finalize prefill。这样老请求的 decode 结果优先进入状态机；
        # 新 prefill 请求仍然只会在下一轮 decode。
        if pending.decode is not None:
            next_tokens = self.runner.decode_batch_finalize(pending.decode.handle)
            if len(next_tokens) != len(pending.decode.reqs):
                raise RuntimeError(
                    f"runner returned {len(next_tokens)} tokens for "
                    f"{len(pending.decode.reqs)} reqs"
                )
            for req, token in zip(pending.decode.reqs, next_tokens, strict=True):
                req.append_output(token)
                if req.maybe_finish():
                    self._finish_req(req, finished)
                else:
                    self._emit("overlap_decode_done", rid=req.rid, next_token=token)

        for pending_prefill in pending.prefills:
            req = pending_prefill.req
            token = self.runner.prefill_finalize(pending_prefill.handle)
            req.append_output(token)
            if req.maybe_finish():
                self._finish_req(req, finished)
            else:
                req.mark_running()
                self.running_reqs.append(req)
                self._emit("overlap_prefill_done", rid=req.rid, next_token=token)
