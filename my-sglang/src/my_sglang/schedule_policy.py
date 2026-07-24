from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from my_sglang.models import Req
from my_sglang.pools import BaseTokenToKVPoolAllocator
from my_sglang.schedule_batch import MiniScheduleBatch


class AddReqResult(str, Enum):
    """PrefillAdder 对单个候选请求给出的准入结论。

    这个枚举描述的是“本轮 scheduler 应如何处理请求”，不是请求最终的
    生命周期状态。比如 ``CHUNK`` 后请求仍未生成输出 token，只是 prompt
    先完成一段；下一轮它会作为 ``chunked_req`` 再次参与决策。

    ``str`` 让枚举值可以直接用于 JSON trace，例如 ``"admit"``，而不需要
    额外把 Enum 转成字符串。
    """

    # 本轮可以一次填完剩余 prompt（也包括完整 cache hit）。scheduler 会把
    # 请求放入 EXTEND batch；forward 后它会进入 RUNNING / decode，或因达到
    # eos / length 直接结束。
    ADMIT = "admit"

    # 本轮只填到 target_fill_len，prompt 还未结束。scheduler 记录它为唯一
    # chunked_req；已完成的页可入 radix cache，下一轮继续填剩余部分。
    CHUNK = "chunk"

    # 请求本身可运行，但当前 prefill/KV 预算不足。请求留在 waiting_queue，
    # 不释放也不分配新的资源；FCFS 策略会阻止后面的 waiting 请求越过它。
    DEFER = "defer"

    # 请求连最小的一页 KV 都无法放入当前物理池，之后也不可能自行变得可
    # 运行。scheduler 会把它标记为 abort，而不是无限期留在 waiting_queue。
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
        """决定这个请求本轮应完整 prefill、切一段 chunk、等待还是终止。

        先把三个长度区分开；它们混在一起是这个方法最容易读错的地方：

        ``prefix_len``
            已经有 KV slot 的 token 数。新请求来自 radix cache 命中；续
            chunk 则来自该请求自己上一轮已经 committed 的 token。
        ``total_len``
            prompt 的总 token 数，即本轮最终希望填到的位置。
        ``target``
            本轮实际填到的位置。``target == total_len`` 是完整 prefill；
            ``target < total_len`` 是 chunked prefill。

        决策优先级如下。前面的成功即返回，后面的分支不会再执行：

        ``cache hit -> 固定大小 chunk -> 完整 prompt -> 防饿兜底
        -> 缩小 chunk 重试 -> ABORT / DEFER``

        这里同时受两类预算限制：

        * ``remaining_prefill_tokens``：吞吐限制，防止一轮 prefill 吃掉
          太多 prompt token；
        * ``remaining_tokens``：KV 容量限制，已经扣除了为 running decode
          请求预留的空间。

        因此“prompt 还没填完”不等于一定要 ``DEFER``：只要能安全地填一
        段，就返回 ``CHUNK``，下一轮再继续。
        """
        # 1. 先确定本次从 prompt 的哪里开始填。
        #
        # 新请求可以复用 radix cache 的公共前缀；续 chunk 已经持有自己的
        # KV row，不能重新 match，否则会把“本请求已计算的部分”和“共享
        # cache 前缀”混为一谈。
        total_len = len(req.fill_ids)
        prefix_len = req.kv_committed_len if continuing else self._match_len(req)

        # 全 prompt 都已有 KV slot（常见于完整 radix hit）。虽然本轮
        # extend 长度为 0，仍返回 ADMIT：scheduler 还需要把它放进 batch，
        # 运行 forward 以取得下一 token。
        if prefix_len >= total_len:
            return AdmissionDecision(req, AddReqResult.ADMIT, prefix_len, total_len)

        # ``full_extend`` 是尚未映射 KV slot 的 prompt token 数。
        full_extend = total_len - prefix_len

        # 2. 首选“配置的固定 chunk 大小”。
        #
        # 例如 remaining prompt 是 100 token，chunk size 是 32，则优先问：
        # “本轮能否填到 prefix_len + 32？”而不是一开始就尝试完整 100。
        # 这使长 prompt 不会独占一个 prefill batch。
        if (
            self.chunked_prefill_size is not None
            and full_extend > self.chunked_prefill_size
        ):
            chunk_len = self.chunked_prefill_size
            # chunk 不得超过本轮剩余的 prompt 吞吐预算。这里只缩短，不会
            # 直接 DEFER；后面仍会用 KV 容量预算判断缩短后的 chunk 能否放下。
            if self.budget.remaining_prefill_tokens < chunk_len:
                chunk_len = self.budget.remaining_prefill_tokens
            if chunk_len > 0:
                target = prefix_len + chunk_len
                cost = self._candidate_cost(req, prefix_len, target, is_last=False)
                # 非最后一段不会产生 output token，因此 is_last=False 时 cost
                # 不包含 decode 输出预留；但仍包含 extend page 和安全余量。
                if cost <= self.budget.remaining_tokens:
                    return AdmissionDecision(
                        req, AddReqResult.CHUNK, prefix_len, target
                    )

        # 3. 固定 chunk 不可用时，尝试一次把剩余 prompt 全填完。
        #
        # 完整 prefill 要额外为之后的 decode 预留 output token；因此它的
        # cost 通常大于上面的中间 chunk。
        full_cost = self._candidate_cost(req, prefix_len, total_len, is_last=True)
        # 不能用 remaining_prefill_tokens 判断“首请求”：full cache hit 的 extend
        # 长度为 0，不会消耗该预算，却仍然已经占用了一个 request row/输出预留。
        first_admission = self.num_accepted == 0
        throughput_ok = (
            full_extend <= self.budget.remaining_prefill_tokens or first_admission
        )
        if full_cost <= self.budget.remaining_tokens and throughput_ok:
            return AdmissionDecision(req, AddReqResult.ADMIT, prefix_len, total_len)

        # 4. 常规预算太保守时的防饿兜底。
        #
        # ``remaining_tokens`` 已扣除了 running decode 的预留，适合多请求共存；
        # 但在空系统的首请求、或正在续跑的 chunk 上，若严格遵守它可能出现：
        # “物理 KV 明明放得下，却永远没有请求能启动/继续”的死锁式 defer。
        # 此处改为只看真实可用物理容量（free + 可淘汰 cache）。
        physical_tokens = self.allocator.available_size + (
            self.tree_cache.evictable_size() if self.tree_cache else 0
        )
        if first_admission and (continuing or not self.has_running_reqs):
            # 空系统的首请求，或已经开始的 chunk，不能因为“未来
            # decode 预留 + 对齐余量”而永久 defer。只要本次真实扩展能
            # 放进物理池就允许运行，后续压力由 evict/retract/abort 闭环处理。
            fallback_extend = (
                # 即使物理容量很大，也遵守单 chunk 的上限；教学版一次只维护
                # 一个未完成 chunked 请求。
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

        # 5. 前面的“固定 chunk”可能因预算不足失败，但更短的一段仍可能放得
        # 下。此处从大到小枚举，拿到“本轮允许的最大 chunk”。
        if self.chunked_prefill_size is not None:
            max_chunk = min(full_extend, self.chunked_prefill_size)
            if not first_admission:
                # 非首请求必须遵守 prefill 吞吐预算；首请求已经在上面的防饿
                # 分支处理，不能让它因吞吐预算为 0 而无法开始。
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

        # 6. 已经没有任何合规方案。先区分“永远不可能”还是“本轮暂时不行”：
        #
        # * 连一页 KV 都装不下：以后也无法推进，ABORT；
        # * 本轮预算被其他请求/预留占用：以后可能释放，DEFER 并保持 FCFS。
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
        # 成本不是单纯的 ``target - prefix_len``，因为真实 KV allocator 按页
        # 分配。比如 page_size=4，extend 1 token 也至少占用 4 个 slot。
        extend_cost = self._round_page(max(target - prefix_len, 0))
        reserve_ratio = 1.0 if req.retracted_stain else self.new_token_ratio
        output_reserve = (
            # 只有 prompt 最后一段才会得到首个 output token 并进入 decode，
            # 因此中间 chunk 不预留 output 空间。
            self._round_page(math.ceil(req.remaining_new_tokens * reserve_ratio))
            if is_last
            else 0
        )
        # 再留一整页安全余量。它吸收 prompt/输出交界处的 page 对齐需求，
        # 避免 admission 刚通过，下一步分配就没有空间。
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
