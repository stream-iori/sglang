from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import TextIO

from my_sglang.models import BatchForward, Req, RequestStatus
from my_sglang.pools import KVPool, ReqPool, ReqToTokenMap
from my_sglang.radix_cache import MiniRadixCache, RadixNode
from my_sglang.runner import RunnerProtocol


@dataclass(frozen=True)
class StepResult:
    # step() 返回本轮实际做了什么，方便测试断言和人类观察。
    prefill_batch: BatchForward | None
    decode_batch: BatchForward | None
    finished_rids: tuple[str, ...]


@dataclass(frozen=True)
class PrefillPlan:
    # PrefillPlan 把“资源申请结果”和“实际 forward 输入”放在一起。
    # 这样 prefill 阶段可以先完成资源规划，再统一构造 BatchForward 和调用 runner。
    req: Req
    req_pool_idx: int
    prefix_slot_ids: tuple[int, ...]
    new_token_ids: tuple[int, ...]
    new_slot_ids: tuple[int, ...]
    chunk_start: int
    # prefix_cache_nodes 是本次命中的 radix cache 节点。
    # 这些节点已经被 pin，必须在请求结束或规划失败时 release。
    prefix_cache_nodes: tuple[RadixNode, ...] = ()
    is_first_chunk: bool = True
    is_last_chunk: bool = True

    @property
    def all_slot_ids(self) -> tuple[int, ...]:
        return (*self.prefix_slot_ids, *self.new_slot_ids)

    @property
    def chunk_end(self) -> int:
        return self.chunk_start + len(self.new_token_ids)


class MiniScheduler:
    # MiniScheduler 是 normal scheduling 的核心。
    # 它维护两个队列：
    # 1. waiting_queue: 新请求，等待 prefill。
    # 2. running_reqs: 已经 prefill 完，等待一轮轮 decode。
    def __init__(
        self,
        runner: RunnerProtocol,
        *,
        max_running_reqs: int = 128,
        max_total_tokens: int = 8192,
        enable_radix_cache: bool = False,
        radix_cache: MiniRadixCache | None = None,
        chunked_prefill_size: int | None = None,
        trace: bool = False,
        trace_file: TextIO | None = None,
    ):
        # runner 是真正执行模型 forward 的对象；可以是 fake runner，也可以是 MLX adapter。
        self.runner = runner
        # 等待 prefill 的新请求；还没有占用 ReqPool/KVPool 资源。
        self.waiting_queue: list[Req] = []
        # chunked prefill 的中间队列：请求已经占用资源，但 prompt 还没完全填完。
        self.prefilling_reqs: list[Req] = []
        # 已完成 prefill、后续每轮 decode 一个 token 的请求。
        self.running_reqs: list[Req] = []
        # 下面三个结构只模拟 SGLang 的资源管理关系，不存真实张量。
        self.req_pool = ReqPool(max_running_reqs)
        self.kv_pool = KVPool(max_total_tokens)
        self.req_to_token = ReqToTokenMap()
        self.radix_cache = radix_cache if radix_cache is not None else (
            MiniRadixCache() if enable_radix_cache else None
        )
        # rid -> 被这个请求借用的 radix cache 节点。
        # 请求活跃期间这些节点不能被 LRU 淘汰；finish 时 release。
        self._radix_cache_pins: dict[str, tuple[RadixNode, ...]] = {}
        self.chunked_prefill_size = (
            chunked_prefill_size
            if chunked_prefill_size is not None and chunked_prefill_size > 0
            else None
        )
        self.trace = trace
        self.trace_file = trace_file or sys.stderr
        self.last_prefill_batch: BatchForward | None = None
        self.last_decode_batch: BatchForward | None = None

    def add_request(self, req: Req) -> None:
        # 新请求只能进入 waiting_queue，不能直接进入 running_reqs。
        # running 必须发生在 prefill 之后。
        if self.req_pool.get(req.rid) is not None or any(
            existing.rid == req.rid for existing in self.waiting_queue
        ):
            raise ValueError(f"duplicate rid {req.rid!r}")
        self.waiting_queue.append(req)
        self._emit("enqueue", rid=req.rid, prompt_len=len(req.origin_input_ids))

    def step(self) -> StepResult:
        # 记录本轮开始时已经在 running 的请求。
        # 这样本轮新 prefill 的请求不会立刻 decode，避免同一步里跑两次模型。
        decode_candidates = list(self.running_reqs)
        finished: list[str] = []
        # normal scheduling 的一轮：先接纳 waiting 做 prefill，再 decode 老的 running。
        prefill_batch = self._run_prefill(finished)
        decode_batch = self._run_decode(decode_candidates, finished)

        # 保留最近一次非空 batch，便于集成测试和调试观察。
        if prefill_batch is not None:
            self.last_prefill_batch = prefill_batch
        if decode_batch is not None:
            self.last_decode_batch = decode_batch
        return StepResult(prefill_batch, decode_batch, tuple(finished))

    def run_until_complete(self) -> list[Req]:
        # 简单阻塞式运行：只要还有 waiting 或 running 请求，就不断 step。
        completed: list[Req] = []
        while self.waiting_queue or self.prefilling_reqs or self.running_reqs:
            # step() 完成后请求可能已从队列移除，所以先用 rid 保存对象引用。
            active_reqs = [
                *self.waiting_queue,
                *self.prefilling_reqs,
                *self.running_reqs,
            ]
            reqs_before_step = {req.rid: req for req in active_reqs}
            result = self.step()
            for rid in result.finished_rids:
                if rid in reqs_before_step:
                    completed.append(reqs_before_step[rid])
        return completed

    def _run_prefill(self, finished: list[str]) -> BatchForward | None:
        if self.chunked_prefill_size is not None:
            return self._run_chunked_prefill(finished)
        return self._run_prefill_waiting(finished)

    def _run_prefill_waiting(self, finished: list[str]) -> BatchForward | None:
        if not self.waiting_queue:
            return None

        # 取出当前所有 waiting 请求，先从 waiting_queue 移除。
        # 如果资源规划失败，会在 except 中把它们原样放回队头。
        reqs = self.waiting_queue
        self.waiting_queue = []
        try:
            planned = self._prepare_prefill_plans(reqs)
        except Exception:
            self.waiting_queue = reqs + self.waiting_queue
            raise

        batch = self._build_prefill_batch(planned)
        self._emit(
            "prefill_batch",
            rids=[req.rid for req in batch.reqs],
            seq_lens=list(batch.seq_lens),
        )

        for plan in planned:
            req = plan.req
            self._attach_prefill_plan(plan)

            # prefill 复用 prefix_slot_ids，只计算 new_token_ids，返回第一个生成 token。
            token = self.runner.prefill(
                req_id=req.rid,
                new_token_ids=list(plan.new_token_ids),
                full_token_ids=list(req.origin_input_ids),
                prefix_slot_ids=list(plan.prefix_slot_ids),
                new_slot_ids=list(plan.new_slot_ids),
                req_pool_idx=plan.req_pool_idx,
            )
            req.append_output(token)
            if req.maybe_finish():
                # 如果 prefill 直接生成 eos，或者 max_new_tokens=1，就不进入 running。
                self._finish_req(req, finished)
            else:
                req.mark_running()
                self.running_reqs.append(req)
                self._emit("prefill_done", rid=req.rid, next_token=token)

        return batch

    def _run_chunked_prefill(self, finished: list[str]) -> BatchForward | None:
        if not self.waiting_queue and not self.prefilling_reqs:
            return None

        # 先推进已经切到一半的长 prompt，再接纳新的 waiting 请求。
        # 这能直观看到 chunked prefill 如何让长 prompt 跨多个 step 前进。
        continuing_reqs = list(self.prefilling_reqs)
        waiting_reqs = self.waiting_queue
        self.waiting_queue = []

        try:
            planned = self._prepare_chunked_prefill_plans(
                continuing_reqs,
                waiting_reqs,
            )
        except Exception:
            self.waiting_queue = waiting_reqs + self.waiting_queue
            raise

        if not planned:
            return None

        batch = self._build_prefill_batch(planned)
        self._emit(
            "chunked_prefill_batch",
            rids=[req.rid for req in batch.reqs],
            chunk_starts=list(batch.chunk_starts_by_req),
            is_last=list(batch.is_last_prefill_chunk_by_req),
        )

        for plan in planned:
            self._attach_prefill_plan(plan)
            token = self._run_prefill_plan(plan)
            self._advance_after_prefill_plan(plan, token, finished)

        return batch

    def _prepare_prefill_plans(self, reqs: list[Req]) -> list[PrefillPlan]:
        # 资源申请阶段要求“全有或全无”：任意请求失败，都回滚本批已申请的资源。
        planned: list[PrefillPlan] = []
        try:
            for req in reqs:
                prefix_cache_nodes: tuple[RadixNode, ...] = ()
                try:
                    (
                        prefix_slot_ids,
                        new_token_ids,
                        prefix_cache_nodes,
                    ) = self._split_cached_prefix(req)
                    req_pool_idx = self.req_pool.alloc(req.rid)
                    new_slot_ids = tuple(self.kv_pool.alloc_many(len(new_token_ids)))
                except Exception:
                    # _split_cached_prefix 可能已经 pin 了 cache prefix。
                    # 后续资源申请失败时，必须 release，避免节点永远不能被 LRU 淘汰。
                    if self.radix_cache is not None:
                        self.radix_cache.release_nodes(prefix_cache_nodes)
                    self.req_pool.free(req.rid)
                    raise

                planned.append(
                    PrefillPlan(
                        req=req,
                        req_pool_idx=req_pool_idx,
                        prefix_slot_ids=prefix_slot_ids,
                        new_token_ids=new_token_ids,
                        new_slot_ids=new_slot_ids,
                        chunk_start=len(prefix_slot_ids),
                        prefix_cache_nodes=prefix_cache_nodes,
                    )
                )
            return planned
        except Exception:
            self._rollback_prefill_plans(planned)
            raise

    def _split_cached_prefix(
        self,
        req: Req,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[RadixNode, ...]]:
        # radix cache 命中的 prefix 直接复用已有 KV slot，只为未命中的 suffix 分配新 slot。
        if self.radix_cache is None:
            return (), tuple(req.origin_input_ids), ()

        # pin=True 表示命中的 prefix slot 将被本请求借用。
        # 在请求 finish 前，LRU 淘汰不能删除这些节点。
        match = self.radix_cache.match_prefix(req.origin_input_ids, pin=True)
        prefix_slot_ids = tuple(match.slot_ids)
        suffix_token_ids = tuple(req.origin_input_ids[len(prefix_slot_ids) :])
        return prefix_slot_ids, suffix_token_ids, match.matched_nodes

    def _rollback_prefill_plans(self, planned: list[PrefillPlan]) -> None:
        for plan in reversed(planned):
            self.kv_pool.free_many(list(plan.new_slot_ids))
            if self.radix_cache is not None:
                self.radix_cache.release_nodes(plan.prefix_cache_nodes)
            if plan.is_first_chunk:
                self.req_to_token.remove_req(plan.req_pool_idx)
                self.req_pool.free(plan.req.rid)
                plan.req.req_pool_idx = None
                plan.req.kv_slots.clear()
                plan.req.prefix_slot_ids.clear()
                plan.req.owned_kv_slots.clear()

    def _build_prefill_batch(self, planned: list[PrefillPlan]) -> BatchForward:
        return BatchForward(
            # BatchForward 只是把调度结果结构化，方便理解和测试。
            mode="prefill",
            reqs=tuple(plan.req for plan in planned),
            input_ids_by_req=tuple(plan.new_token_ids for plan in planned),
            req_pool_indices=tuple(plan.req_pool_idx for plan in planned),
            out_cache_locs=tuple(plan.new_slot_ids for plan in planned),
            seq_lens=tuple(plan.chunk_end for plan in planned),
            prefix_slot_ids_by_req=tuple(plan.prefix_slot_ids for plan in planned),
            chunk_starts_by_req=tuple(plan.chunk_start for plan in planned),
            is_last_prefill_chunk_by_req=tuple(
                plan.is_last_chunk for plan in planned
            ),
        )

    def _attach_prefill_plan(self, plan: PrefillPlan) -> None:
        req = plan.req
        if plan.is_first_chunk:
            # 把资源归属写回 Req，后续 finish 时才能决定释放还是交给 radix cache。
            req.req_pool_idx = plan.req_pool_idx
            req.prefix_slot_ids.extend(plan.prefix_slot_ids)
            req.kv_slots.extend(plan.prefix_slot_ids)
            if plan.prefix_cache_nodes:
                self._radix_cache_pins[req.rid] = plan.prefix_cache_nodes
            for offset, slot in enumerate(plan.prefix_slot_ids):
                self.req_to_token.set(plan.req_pool_idx, offset, slot)

        req.owned_kv_slots.extend(plan.new_slot_ids)
        req.kv_slots.extend(plan.new_slot_ids)
        # 记录本轮 chunk 每个序列位置对应哪个 KV slot。
        for offset, slot in enumerate(plan.new_slot_ids):
            self.req_to_token.set(plan.req_pool_idx, plan.chunk_start + offset, slot)
        req.prefill_pos = plan.chunk_end

    def _prepare_chunked_prefill_plans(
        self,
        continuing_reqs: list[Req],
        waiting_reqs: list[Req],
    ) -> list[PrefillPlan]:
        # 这个方法只负责“规划”，不会修改请求的 prefill_pos，也不会调用模型：
        # 1. 为每个请求确定本轮要处理的 prompt 切片；
        # 2. 申请这个切片所需的 ReqPool/KVPool 资源；
        # 3. 把以上结果封装成 PrefillPlan，交给调用方统一 attach 和 forward。
        #
        # 两类输入的含义不同：
        # - continuing_reqs 已经完成过至少一个 chunk，拥有 req_pool_idx 和历史 KV；
        # - waiting_reqs 尚未开始 prefill，需要申请请求槽，并可能先复用 radix prefix。
        # 返回顺序也就是后续 BatchForward 中的请求顺序：续填请求优先，新请求随后。

        # 只有开启 chunked prefill 时才允许走到这里。
        # 上层 _run_prefill 已根据该配置分流；assert 用来尽早暴露内部调用错误。
        assert self.chunked_prefill_size is not None

        # planned 记录本批已经成功申请资源的计划。
        # 它不仅是返回值，也是异常时进行批量回滚的“事务日志”。
        planned: list[PrefillPlan] = []
        try:
            # 优先推进正在 PREFILLING 的请求，避免一个长 prompt 开始后一直被新请求挤压。
            # continuing_reqs 是 prefilling_reqs 的快照；状态检查可过滤其中已经发生
            # 生命周期变化的陈旧项，防止为非 PREFILLING 请求重复分配 KV slot。
            for req in continuing_reqs:
                if req.status is not RequestStatus.PREFILLING:
                    continue

                # 续填请求复用已有 req_pool_idx，从 req.prefill_pos 开始切下一块，
                # 只为本轮新 token 申请 KV slot；不会重新匹配 radix cache。
                planned.append(self._plan_next_prefill_chunk(req))

            # 新请求从 prompt 的第一个未缓存位置开始规划首块：这里会申请
            # req_pool_idx、pin 命中的 radix 节点，并为未缓存的 chunk 申请 KV slot。
            for req in waiting_reqs:
                planned.append(self._plan_first_prefill_chunk(req))

            # 至此整批资源均申请成功，但尚未写回 Req，也尚未执行模型。
            # 调用方 _run_chunked_prefill 会据此构造 batch，再逐个 attach/forward。
            return planned
        except Exception:
            # 任一请求规划失败时，撤销本批此前所有成功计划，保证“整批全成或全退”：
            # - 首块计划释放新申请的请求槽、KV slot 和 radix pin；
            # - 续填计划只释放本轮新申请的 KV slot，保留此前 chunk 已有的状态和资源。
            # 当前恰好失败的计划尚未 append：首块规划会在自身 except 中清理请求槽
            # 和 radix pin；续填规划只有原子式 KV alloc_many，失败时不会留下部分 slot。
            # 异常继续抛给 _run_chunked_prefill，由上层把 waiting_reqs 放回等待队列。
            self._rollback_prefill_plans(planned)
            raise

    def _plan_first_prefill_chunk(self, req: Req) -> PrefillPlan:
        assert self.chunked_prefill_size is not None
        prefix_cache_nodes: tuple[RadixNode, ...] = ()
        try:
            prefix_slot_ids, _, prefix_cache_nodes = self._split_cached_prefix(req)
            chunk_start = len(prefix_slot_ids)
            chunk_token_ids = tuple(
                req.origin_input_ids[
                    chunk_start : chunk_start + self.chunked_prefill_size
                ]
            )
            req_pool_idx = self.req_pool.alloc(req.rid)
            chunk_slot_ids = tuple(self.kv_pool.alloc_many(len(chunk_token_ids)))
        except Exception:
            self.req_pool.free(req.rid)
            if self.radix_cache is not None:
                self.radix_cache.release_nodes(prefix_cache_nodes)
            raise

        return PrefillPlan(
            req=req,
            req_pool_idx=req_pool_idx,
            prefix_slot_ids=prefix_slot_ids,
            new_token_ids=chunk_token_ids,
            new_slot_ids=chunk_slot_ids,
            chunk_start=chunk_start,
            prefix_cache_nodes=prefix_cache_nodes,
            is_first_chunk=True,
            is_last_chunk=chunk_start + len(chunk_token_ids)
            >= len(req.origin_input_ids),
        )

    def _plan_next_prefill_chunk(self, req: Req) -> PrefillPlan:
        assert self.chunked_prefill_size is not None
        req_pool_idx = self._require_req_pool_idx(req)
        chunk_start = req.prefill_pos
        chunk_token_ids = tuple(
            req.origin_input_ids[chunk_start : chunk_start + self.chunked_prefill_size]
        )
        if not chunk_token_ids:
            raise RuntimeError(f"request {req.rid} has no remaining prefill tokens")
        chunk_slot_ids = tuple(self.kv_pool.alloc_many(len(chunk_token_ids)))
        return PrefillPlan(
            req=req,
            req_pool_idx=req_pool_idx,
            prefix_slot_ids=(),
            new_token_ids=chunk_token_ids,
            new_slot_ids=chunk_slot_ids,
            chunk_start=chunk_start,
            is_first_chunk=False,
            is_last_chunk=chunk_start + len(chunk_token_ids)
            >= len(req.origin_input_ids),
        )

    def _run_prefill_plan(self, plan: PrefillPlan) -> int:
        req = plan.req
        if plan.is_first_chunk:
            return self.runner.prefill(
                req_id=req.rid,
                new_token_ids=list(plan.new_token_ids),
                full_token_ids=list(req.origin_input_ids[: plan.chunk_end]),
                prefix_slot_ids=list(plan.prefix_slot_ids),
                new_slot_ids=list(plan.new_slot_ids),
                req_pool_idx=plan.req_pool_idx,
            )

        return self.runner.extend(
            req_id=req.rid,
            new_token_ids=list(plan.new_token_ids),
            new_slot_ids=list(plan.new_slot_ids),
        )

    def _advance_after_prefill_plan(
        self,
        plan: PrefillPlan,
        token: int,
        finished: list[str],
    ) -> None:
        req = plan.req
        if not plan.is_last_chunk:
            req.status = RequestStatus.PREFILLING
            if req not in self.prefilling_reqs:
                self.prefilling_reqs.append(req)
            self._emit(
                "prefill_chunk_done",
                rid=req.rid,
                prefill_pos=req.prefill_pos,
                ignored_token=token,
            )
            return

        if req in self.prefilling_reqs:
            self.prefilling_reqs.remove(req)
        req.append_output(token)
        if req.maybe_finish():
            self._finish_req(req, finished)
        else:
            req.mark_running()
            self.running_reqs.append(req)
            self._emit("prefill_done", rid=req.rid, next_token=token)

    def _run_decode(
        self,
        candidates: list[Req],
        finished: list[str],
    ) -> BatchForward | None:
        decode_reqs = self._collect_decode_reqs(candidates)
        if not decode_reqs:
            return None

        # decode 每个请求只需要为“本轮输入 token”申请一个新的 KV slot。
        allocated: list[tuple[Req, int]] = []
        try:
            slots = self.kv_pool.alloc_many(len(decode_reqs))
            allocated = list(zip(decode_reqs, slots, strict=True))
        except Exception:
            self.kv_pool.free_many([slot for _, slot in allocated])
            raise

        batch = self._build_decode_batch(decode_reqs, allocated)
        self._emit(
            "decode_batch",
            rids=[req.rid for req in batch.reqs],
            input_ids=[ids[0] for ids in batch.input_ids_by_req],
        )

        for req, slot in allocated:
            req_pool_idx = self._require_req_pool_idx(req)
            req.kv_slots.append(slot)
            req.owned_kv_slots.append(slot)
            # 这里记录的是 decode 输入 token 的 KV 位置。
            # append_output 发生在后面，所以当前 len(full_token_ids)-1 正好是输入 token 的序列位置。
            self.req_to_token.set(req_pool_idx, len(req.full_token_ids) - 1, slot)

        # 真正批量调用模型：多个 request id 一起进入 runner.decode_batch。
        next_tokens = self.runner.decode_batch([req.rid for req in decode_reqs])
        if len(next_tokens) != len(decode_reqs):
            raise RuntimeError(
                f"runner returned {len(next_tokens)} tokens for {len(decode_reqs)} reqs"
            )

        for req, token in zip(decode_reqs, next_tokens, strict=True):
            # runner 返回的 token 是下一轮 decode 的输入，也会被追加到 output_ids。
            req.append_output(token)
            if req.maybe_finish():
                self._finish_req(req, finished)
            else:
                self._emit("decode_done", rid=req.rid, next_token=token)

        return batch

    def _collect_decode_reqs(self, candidates: list[Req]) -> list[Req]:
        # candidates 是 step 开始时的快照；还要确认请求没有在本轮 prefill 中提前结束。
        return [
            req
            for req in candidates
            if req.status is RequestStatus.RUNNING and req in self.running_reqs
        ]

    def _build_decode_batch(
        self,
        decode_reqs: list[Req],
        allocated: list[tuple[Req, int]],
    ) -> BatchForward:
        return BatchForward(
            mode="decode",
            reqs=tuple(decode_reqs),
            # decode 阶段输入的是每个请求当前最后一个 token。
            input_ids_by_req=tuple((req.last_token_id,) for req in decode_reqs),
            req_pool_indices=tuple(
                self._require_req_pool_idx(req) for req in decode_reqs
            ),
            out_cache_locs=tuple((slot,) for _, slot in allocated),
            seq_lens=tuple(len(req.full_token_ids) for req in decode_reqs),
        )

    def _finish_req(self, req: Req, finished: list[str]) -> None:
        # 统一释放请求相关资源：running 队列、req_to_token 映射、ReqPool、KVPool、runner 内部状态。
        req.status = RequestStatus.FINISHED
        if req in self.prefilling_reqs:
            self.prefilling_reqs.remove(req)
        if req in self.running_reqs:
            self.running_reqs.remove(req)
        if req.req_pool_idx is not None:
            self.req_to_token.remove_req(req.req_pool_idx)
        self.req_pool.free(req.rid)
        self._cache_or_free_req_slots(req)
        req.kv_slots.clear()
        req.prefix_slot_ids.clear()
        req.owned_kv_slots.clear()
        req.prefill_pos = 0
        self.runner.remove_request(req.rid)
        finished.append(req.rid)
        self._emit("finish", rid=req.rid, reason=req.finish_reason)

    def _cache_or_free_req_slots(self, req: Req) -> None:
        if self.radix_cache is None:
            self.kv_pool.free_many(req.kv_slots)
            return

        # 先 release 本请求借用的 cache prefix。
        # 请求已经 finish 且 req_to_token 已删除，这些 prefix slot 不再被它使用。
        prefix_cache_nodes = self._radix_cache_pins.pop(req.rid, ())
        self.radix_cache.release_nodes(prefix_cache_nodes)

        # full_token_ids 比 kv_slots 可能多一个“刚生成但尚未作为 decode 输入写入”的 token。
        # 只有已经拥有 KV slot 的前缀才能进入 radix cache。
        cacheable_len = min(len(req.kv_slots), len(req.full_token_ids))
        cacheable_token_ids = req.full_token_ids[:cacheable_len]
        cacheable_slot_ids = req.kv_slots[:cacheable_len]
        cache_owned_slots: set[int] = set()

        if cacheable_token_ids:
            insert_result = self.radix_cache.insert(
                cacheable_token_ids,
                cacheable_slot_ids,
            )
            # insert_result 只返回新纳入 cache 的 slot；已存在的 prefix slot 本来就归 cache 所有。
            cache_owned_slots.update(insert_result.inserted_slots)
            # insert 可能因为容量上限触发 LRU 淘汰。
            # radix cache 只返回 slot id，真正释放 KVPool 在 scheduler 做。
            self.kv_pool.free_many(list(insert_result.evicted_slots))

        # owned_kv_slots 中没被 cache 接管的 slot 可以释放；prefix_slot_ids 是借来的，不在这里释放。
        releasable_slots = [
            slot for slot in req.owned_kv_slots if slot not in cache_owned_slots
        ]
        self.kv_pool.free_many(releasable_slots)

    def _require_req_pool_idx(self, req: Req) -> int:
        # 类型保护：对已经进入模型执行阶段的请求，req_pool_idx 必须存在。
        if req.req_pool_idx is None:
            raise RuntimeError(f"request {req.rid} has no req_pool_idx")
        return req.req_pool_idx

    def _emit(self, event: str, **payload) -> None:
        # trace 使用 JSON lines，方便人读，也方便后续脚本分析调度过程。
        if not self.trace:
            return
        print(
            json.dumps({"event": event, **payload}, sort_keys=True),
            file=self.trace_file,
        )
