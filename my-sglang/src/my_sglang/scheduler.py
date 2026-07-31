"""单进程教学调度器。

主循环每步只做一种 forward：先尝试 EXTEND（prefill），没有新请求可接纳时
再做 DECODE。请求资源的主线是 ``waiting -> batch -> running -> finished``。
"""

import json
import sys
from dataclasses import dataclass
from typing import Any, TextIO

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
    """一次 scheduler step 对调用者可见的结果和状态事件。"""

    batch: BatchForward | None  # 本轮执行的 forward；纯管理轮次为 None。
    finished_rids: tuple[str, ...] = ()  # 本轮正常结束的请求。
    retracted_rids: tuple[str, ...] = ()  # 因 KV 压力退回 waiting 的请求。
    aborted_rids: tuple[str, ...] = ()  # 本轮不可恢复地终止的请求。
    memory: MemorySnapshot | None = None  # 本轮结束时的内存快照。

    @property
    def prefill_batch(self) -> BatchForward | None:
        """若本轮是 EXTEND，返回该 batch。"""
        return (
            self.batch if self.batch and self.batch.mode is ForwardMode.EXTEND else None
        )

    @property
    def decode_batch(self) -> BatchForward | None:
        """若本轮是 DECODE，返回该 batch。"""
        return (
            self.batch if self.batch and self.batch.mode is ForwardMode.DECODE else None
        )


class MiniScheduler:
    """SGLang-like 单进程调度器：prefill 优先，否则 decode。

    三个长期容器的分工：``waiting_queue`` 等待准入，``chunked_req`` 保存
    唯一的未完成 prompt，``running_batch`` 保存下一轮可 decode 的请求。
    ``last_batch`` 只是上一轮结果到下一轮的交接站。
    """

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
        trace: bool = True,
        trace_file: TextIO | None = None,
    ):
        if not 0 <= new_token_ratio <= 1:
            raise ValueError("new_token_ratio must be between 0 and 1")

        # 执行后端与两级 KV 索引结构。
        self.runner = runner
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

        # 请求生命周期容器。
        self.waiting_queue: list[Req] = []
        self.running_batch = MiniScheduleBatch.empty(
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
        )
        self.last_batch: MiniScheduleBatch | None = None
        self.chunked_req: Req | None = None

        # Admission 配置。
        self.max_prefill_tokens = max_prefill_tokens or max_total_tokens
        self.chunked_prefill_size = (
            chunked_prefill_size
            if chunked_prefill_size is not None and chunked_prefill_size > 0
            else None
        )
        self.new_token_ratio = new_token_ratio

        # 可观察状态：trace、最近一次 batch 快照和预算。
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
        """返回当前唯一的 chunked-prefill 请求。"""
        return [self.chunked_req] if self.chunked_req is not None else []

    @property
    def running_reqs(self) -> list[Req]:
        """返回已完成 prompt、等待 decode 的请求。"""
        return self.running_batch.reqs

    def add_request(self, req: Req) -> None:
        """校验 rid 唯一后，将新请求加入 FCFS 等待队列。"""

        active = self._all_active_reqs()
        if any(existing.rid == req.rid for existing in active):
            raise ValueError(f"duplicate rid {req.rid!r}")

        self.waiting_queue.append(req)
        self._emit("enqueue", rid=req.rid, prompt_len=len(req.origin_input_ids))

    def step(self) -> StepResult:
        """结算上一批，并执行最多一个新的 EXTEND 或 DECODE batch。"""

        finished: list[str] = []
        retracted: list[str] = []
        aborted: list[str] = []

        # 阶段 1：把上一轮仍存活的请求交接到 running_batch。
        self._settle_last_batch()

        # 阶段 2：prefill 优先；没有可接纳请求时才做 decode。
        batch = self._get_new_prefill_batch(aborted)
        if batch is None:
            batch = self._get_decode_batch(retracted, aborted)

        # 可能只发生了 abort/retract 管理事件，没有模型 forward。
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

        # 阶段 3：运行模型。prepare 已预分配 KV，成功才 commit，失败则回滚。
        forward_snapshot = batch.to_forward_batch()
        try:
            tokens = self._run_batch_sync(batch)
            batch.commit_allocated()
        except Exception:
            self._rollback_failed_batch(batch)
            raise

        # 阶段 4：把 token 写回 Req，并把 batch 留给下一轮 settle。
        self._process_batch_result(batch, tokens, finished)
        self.last_batch = batch
        if batch.forward_mode is ForwardMode.DECODE:
            self.running_batch = MiniScheduleBatch.empty(
                self.req_to_token_pool,
                self.token_to_kv_pool_allocator,
                self.tree_cache,
            )

        if batch.forward_mode is ForwardMode.EXTEND:
            self.last_prefill_batch = forward_snapshot
        else:
            self.last_decode_batch = forward_snapshot
        self.assert_consistent()
        return StepResult(
            forward_snapshot,
            tuple(finished),
            tuple(retracted),
            tuple(aborted),
            self.memory_snapshot(),
        )

    def run_until_complete(self) -> list[Req]:
        """持续 step，直到所有请求正常结束或 abort。"""
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
        """过滤上一轮结果，并恢复下一轮的 decode 候选集合。"""

        if self.last_batch is None:
            return
        batch = self.last_batch
        if batch.forward_mode is ForwardMode.EXTEND:
            # 未完成 prompt 的 chunked_req 由独立字段持有，不能提前 decode。
            exclude = {self.chunked_req} if self.chunked_req is not None else set()
            batch.filter_batch(exclude=exclude)
            if not batch.is_empty():
                self.running_batch.merge_batch(batch)
        else:
            # DECODE 本来就使用 running_batch 对象；过滤结束请求后把对象交还。
            batch.filter_batch()
            self.running_batch = batch
        self.last_batch = None

    def _get_new_prefill_batch(self, aborted: list[str]) -> MiniScheduleBatch | None:
        """为本轮选择请求，并准备一个可直接 forward 的 EXTEND batch。"""

        if not self.waiting_queue and self.chunked_req is None:
            return None

        # 阶段 1：建立一次性预算并进行纯决策。chunked_req 已经占有一行，
        # 因此可接纳数量要把这条现有行加回来。
        max_new_reqs = self.req_to_token_pool.available_size + (
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
            self.waiting_queue,
            self.chunked_req,
            max_new_reqs=max_new_reqs,
        )

        # 阶段 2：执行 ABORT 决策，留下真正需要进入 batch 的决定。
        accepted_decisions = self._apply_prefill_decisions(decisions, aborted)
        if not accepted_decisions:
            return None

        # 阶段 3：绑定请求行/cache 前缀，分配本轮新增 KV slot。
        batch = self._prepare_prefill_batch(accepted_decisions)

        # 只有 batch 完整准备成功后才从 waiting 删除，保证失败时请求不会丢失。
        accepted_reqs = {decision.req for decision in accepted_decisions}
        self.waiting_queue = [
            req for req in self.waiting_queue if req not in accepted_reqs
        ]
        self._emit(
            "prefill_admit",
            rids=[req.rid for req in batch.reqs],
            decisions=[
                decision.result.value for decision in accepted_decisions
            ],
            remaining_tokens=adder.budget.remaining_tokens,
        )
        return batch

    def _apply_prefill_decisions(
        self,
        decisions: list[AdmissionDecision],
        aborted: list[str],
    ) -> list[AdmissionDecision]:
        """执行终止决策，并返回本轮应进入 EXTEND batch 的决定。"""

        accepted_decisions: list[AdmissionDecision] = []
        for decision in decisions:
            if decision.is_accepted:
                accepted_decisions.append(decision)
                continue
            if decision.result is AddReqResult.ABORT:
                reason = decision.reason or "admission failed"
                self._abort_waiting(decision.req, reason)
                aborted.append(decision.req.rid)
        return accepted_decisions

    def _prepare_prefill_batch(
        self, accepted_decisions: list[AdmissionDecision]
    ) -> MiniScheduleBatch:
        """将已接纳决定落成请求行、KV 分配和 batch 元数据。"""

        newly_attached: list[Req] = []
        first_extend_flags: list[bool] = []
        try:
            for decision in accepted_decisions:
                req = decision.req
                continuing = req is self.chunked_req
                if not continuing:
                    self._attach_new_request(req)
                    newly_attached.append(req)
                first_extend_flags.append(not continuing)
                req.fill_len = decision.target_fill_len
                req.extend_input_len = req.fill_len - req.kv_allocated_len

            accepted_reqs = [decision.req for decision in accepted_decisions]
            needed_pages = self._extend_pages_needed(accepted_reqs)
            self._evict_for_pages(needed_pages)
            batch = MiniScheduleBatch(
                reqs=accepted_reqs,
                forward_mode=ForwardMode.EXTEND,
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                tree_cache=self.tree_cache,
                chunked_req=self.chunked_req,
                first_extend_by_req=tuple(first_extend_flags),
            )
            batch.prepare_for_extend()
        except Exception:
            # 只解绑本方法刚 attach 的请求；原 chunked_req 保留原有绑定。
            for req in reversed(newly_attached):
                self._release_active_memory(req, keep_cache=True)
                req.reset_for_retract()
            raise
        return batch

    def _get_decode_batch(
        self, retracted: list[str], aborted: list[str]
    ) -> MiniScheduleBatch | None:
        """为所有 RUNNING 请求准备 DECODE；内存不足时逐步降压。"""

        decode_reqs = [
            req
            for req in self.running_batch.reqs
            if req.status is RequestStatus.RUNNING
        ]
        if not decode_reqs:
            return None

        # 降压顺序固定为：淘汰 cache -> retract 部分请求 -> 最后一个也
        # 无法运行则 abort。每轮缩小 decode_reqs，直到物理页足够。
        while decode_reqs:
            needed_pages = self._decode_pages_needed(decode_reqs)
            self._evict_for_pages(needed_pages)
            if needed_pages <= self._free_page_count():
                break

            if len(decode_reqs) == 1:
                req = decode_reqs.pop()
                self._abort_active(req, "KV OOM after cache eviction and retraction")
                aborted.append(req.rid)
                break

            victim = self._choose_retract_victim(decode_reqs)
            decode_reqs.remove(victim)
            self._retract_req(victim)
            retracted.append(victim.rid)
            self._emit("retract", rid=victim.rid)

        if not decode_reqs:
            self.running_batch.reqs = []
            return None

        self.running_batch.reqs = decode_reqs
        self.running_batch.forward_mode = ForwardMode.DECODE
        self.running_batch.prepare_for_decode()
        return self.running_batch

    @staticmethod
    def _choose_retract_victim(reqs: list[Req]) -> Req:
        """优先 retract 生成进度较少、当前上下文较长的请求。"""

        return min(
            reqs,
            key=lambda req: (req.generated_count, -len(req.full_token_ids)),
        )

    def _run_batch_sync(self, batch: MiniScheduleBatch) -> list[int]:
        """同步执行一个已 prepare 的 batch，并统一返回 Python ``int``。"""

        if batch.forward_mode is ForwardMode.DECODE:
            tokens = self.runner.decode_batch([req.rid for req in batch.reqs])
            if len(tokens) != len(batch.reqs):
                raise RuntimeError("runner returned wrong decode batch size")
            return [int(token) for token in tokens]

        return self._run_extend_batch_sync(batch)

    def _run_extend_batch_sync(self, batch: MiniScheduleBatch) -> list[int]:
        """逐请求调用首次 prefill 或后续 chunk extend 接口。"""

        tokens: list[int] = []
        for index, req in enumerate(batch.reqs):
            new_token_ids = list(batch.input_ids_by_req[index])
            new_slot_ids = [
                int(slot) for slot in batch.out_cache_locs_by_req[index]
            ]
            if batch.first_extend_by_req[index]:
                token = self.runner.prefill(
                    req_id=req.rid,
                    new_token_ids=new_token_ids,
                    full_token_ids=list(req.fill_ids[: req.fill_len]),
                    prefix_slot_ids=[int(x) for x in req.prefix_indices],
                    new_slot_ids=new_slot_ids,
                    req_pool_idx=self._require_req_pool_idx(req),
                )
            else:
                token = self.runner.extend(
                    req_id=req.rid,
                    new_token_ids=new_token_ids,
                    new_slot_ids=new_slot_ids,
                )
            tokens.append(int(token))
        return tokens

    def _process_batch_result(
        self, batch: MiniScheduleBatch, tokens: list[int], finished: list[str]
    ) -> None:
        """按 batch 模式把模型输出提交到请求生命周期。"""

        if batch.forward_mode is ForwardMode.DECODE:
            self._process_decode_result(batch, tokens, finished)
            return

        self._process_extend_result(batch, tokens, finished)

    def _process_decode_result(
        self, batch: MiniScheduleBatch, tokens: list[int], finished: list[str]
    ) -> None:
        """追加每个 decode token，并结束达到 EOS/长度上限的请求。"""

        for req, token in zip(batch.reqs, tokens, strict=True):
            req.append_output(token)
            if req.maybe_finish():
                self._finish_req(req, finished)

    def _process_extend_result(
        self, batch: MiniScheduleBatch, tokens: list[int], finished: list[str]
    ) -> None:
        """处理中间 chunk，或提交完整 prefill 产生的首个输出 token。"""

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
        """给新请求分配 request row，并绑定已命中的 radix KV 前缀。

        本方法只建立已有前缀的所有权和映射；未命中的 suffix 由
        ``MiniScheduleBatch.prepare_for_extend`` 分配。
        """

        # 先查询并 pin cache；请求存活期间命中节点不能被淘汰。
        match = (
            self.tree_cache.match_prefix(req.fill_ids, pin=True)
            if self.tree_cache is not None
            else None
        )
        try:
            req.req_pool_idx = self.req_to_token_pool.alloc_one(req)
        except Exception:
            # row 分配失败时撤销刚才的 pin，避免 cache 永久不可淘汰。
            if match is not None and match.token_count:
                assert self.tree_cache is not None
                self.tree_cache.dec_lock_ref(match.last_node)
            raise
        if match is not None:
            req.prefix_indices = np.asarray(match.slot_ids, dtype=np.int64)
            req.last_node = match.last_node if match.token_count else None
            req.cache_protected_len = match.token_count
        prefix_len = len(req.prefix_indices)
        if prefix_len:
            # 把逻辑 prompt 位置映射到已存在的物理 cache slot。
            self.req_to_token_pool.write(req.req_pool_idx, 0, req.prefix_indices)

        # cache KV 已经真实计算完成，因此同时是 allocated 和 committed。
        req.kv_allocated_len = prefix_len
        req.kv_committed_len = prefix_len

    def _cache_unfinished_req(self, req: Req) -> None:
        """将 chunked-prefill 已提交的完整页交给 radix cache 托管。"""

        # 只把已 committed 的完整 page 交给 radix cache；未满页的尾部
        # 继续归请求所有。insert 可能返回已有 prefix 的 slot，因此要
        # 重写 req_to_token，并只释放不与新 prefix 共页的重复 page。
        if self.tree_cache is None or req.req_pool_idx is None:
            return
        cacheable_len = (
            req.kv_committed_len
            // self.tree_cache.page_size
            * self.tree_cache.page_size
        )
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
        """完成请求、释放物理资源，并记录本轮 finish 事件。"""
        req.status = RequestStatus.FINISHED
        self._release_active_memory(req, keep_cache=False)
        self.runner.remove_request(req.rid)
        finished.append(req.rid)
        self._emit("finish", rid=req.rid, reason=req.finish_reason)

    def _release_active_memory(self, req: Req, *, keep_cache: bool) -> None:
        """释放请求行和私有 KV page，可选保留已缓存的公共前缀。"""

        if req.req_pool_idx is None:
            return
        all_slots = self.req_to_token_pool.row(req.req_pool_idx, req.kv_allocated_len)
        protected_slots = self._cache_slots_to_keep_on_release(
            req, all_slots, keep_cache=keep_cache
        )

        # 只释放不与 cache 保护区共页的 page，然后归还 request row。
        self.token_to_kv_pool_allocator.free_unshared_pages(all_slots, protected_slots)
        self.req_to_token_pool.free(req)
        self._clear_req_memory_state(req)

    def _cache_slots_to_keep_on_release(
        self,
        req: Req,
        all_slots: np.ndarray,
        *,
        keep_cache: bool,
    ) -> np.ndarray:
        """解除 cache pin，并返回释放请求时仍需保留的 cache slot。"""

        if self.tree_cache is None:
            return np.empty((0,), dtype=np.int64)

        if req.last_node is not None:
            self.tree_cache.dec_lock_ref(req.last_node)
            req.last_node = None

        if keep_cache:
            return req.prefix_indices

        # 正常结束时，将所有已提交的完整页写入 cache，再查询最终受保护 slot。
        cacheable_len = (
            req.kv_committed_len
            // self.tree_cache.page_size
            * self.tree_cache.page_size
        )
        result = self.tree_cache.insert(
            req.full_token_ids[:cacheable_len], all_slots[:cacheable_len]
        )
        self.token_to_kv_pool_allocator.free(result.evicted_slots)
        match = self.tree_cache.match_prefix(
            req.full_token_ids[:cacheable_len], pin=False
        )
        return np.asarray(match.slot_ids, dtype=np.int64)

    @staticmethod
    def _clear_req_memory_state(req: Req) -> None:
        """清空 Req 上所有 request row / KV 相关字段。"""

        req.req_pool_idx = None
        req.prefix_indices = np.empty((0,), dtype=np.int64)
        req.last_node = None
        req.cache_protected_len = 0
        req.kv_allocated_len = 0
        req.kv_committed_len = 0
        req.fill_len = 0
        req.extend_input_len = 0

    def _retract_req(self, req: Req) -> None:
        """释放物理状态但保留 output_ids，将请求放回 waiting。"""
        # retract 保留 output_ids（逻辑进度），但丢弃 row/runner/非缓存 KV。
        # retracted_stain 会让下次 admission 按全部剩余输出做保守预留。
        self._release_active_memory(req, keep_cache=True)
        self.runner.remove_request(req.rid)
        req.reset_for_retract()
        self.waiting_queue.append(req)

    def _abort_active(self, req: Req, message: str) -> None:
        """释放一个 active 请求并以 abort 结束。"""
        self._release_active_memory(req, keep_cache=True)
        self.runner.remove_request(req.rid)
        req.mark_aborted(message)

    def _abort_waiting(self, req: Req, message: str) -> None:
        """从 waiting 删除请求并以 abort 结束。"""
        if req in self.waiting_queue:
            self.waiting_queue.remove(req)
        req.mark_aborted(message)
        self._emit("abort", rid=req.rid, reason=message)

    def _rollback_failed_batch(self, batch: MiniScheduleBatch) -> None:
        """撤销失败 batch 的未提交分配，并让请求重新等待。"""
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
        """计算一组 EXTEND 请求还需申请多少新 page。"""

        total = 0
        for req in reqs:
            last = self._last_loc(req)
            total += self.token_to_kv_pool_allocator.required_pages_for_extend(
                req.kv_allocated_len, req.fill_len, last
            )
        return total

    def _decode_pages_needed(self, reqs: list[Req]) -> int:
        """计算一个 DECODE batch 还需申请多少新 page。"""
        return self.token_to_kv_pool_allocator.required_pages_for_decode(
            [req.kv_allocated_len + 1 for req in reqs],
            [self._last_loc(req) for req in reqs],
        )

    def _evict_for_pages(self, needed_pages: int) -> None:
        """必要时淘汰未锁定 radix cache，尽量凑足目标页数。"""
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
        """返回 allocator 当前空闲整页数。"""
        return (
            self.token_to_kv_pool_allocator.available_size
            // self.token_to_kv_pool_allocator.page_size
        )

    def _last_loc(self, req: Req) -> int:
        """返回请求最后一个已分配 token 对应的物理 KV slot。"""

        if req.req_pool_idx is None or req.kv_allocated_len == 0:
            return -1
        last_slot = self.req_to_token_pool.get(
            req.req_pool_idx, req.kv_allocated_len - 1
        )
        if last_slot is None:
            raise RuntimeError(f"request {req.rid} has an unmapped KV tail")
        return last_slot

    def memory_snapshot(self) -> MemorySnapshot:
        """读取当前 pool/cache/admission 的汇总账本。"""
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
        """集中检查 request row、KV page 和请求边界不变量。"""
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
        """是否还有 waiting、chunked、running 或待 settle 的工作。"""
        return bool(
            self.waiting_queue
            or self.chunked_req is not None
            or not self.running_batch.is_empty()
            or self.last_batch is not None
        )

    def _all_active_reqs(self) -> list[Req]:
        """按首次出现顺序返回所有容器中的去重请求。"""
        reqs = [*self.waiting_queue, *self.running_batch.reqs]
        if self.last_batch is not None:
            reqs.extend(self.last_batch.reqs)
        if self.chunked_req is not None:
            reqs.append(self.chunked_req)
        # dict 保序去重；Req 采用 identity hash。
        return list(dict.fromkeys(reqs))

    @staticmethod
    def _require_req_pool_idx(req: Req) -> int:
        """读取已绑定请求行；未绑定说明调用阶段错误。"""
        if req.req_pool_idx is None:
            raise RuntimeError(f"request {req.rid} has no req_pool_idx")
        return req.req_pool_idx

    def _emit(self, event: str, **payload: Any) -> None:
        """按 JSON Lines 格式输出一个可选 trace 事件。"""
        if not self.trace:
            return
        print(
            json.dumps({"event": event, **payload}, sort_keys=True),
            file=self.trace_file,
        )
