"""Prefill admission：决定请求本轮完整运行、切 chunk、等待或终止。

本模块只做预算决策，不申请请求行或 KV slot。真正的资源绑定发生在
``MiniScheduler._get_new_prefill_batch()``。
"""

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
    """PrefillAdder 对一个候选请求作出的单轮准入决定。

    该对象只描述“本轮是否接纳、以及 prompt 填到哪里”，不保存真实的
    KV slot，也不会直接修改请求状态。scheduler 只会把 ``ADMIT`` 和
    ``CHUNK`` 对应的请求放进 EXTEND batch；``DEFER`` 留在等待队列，
    ``ABORT`` 则终止请求。

    两个长度字段都是 ``req.fill_ids`` 上的绝对 token 边界，采用左闭右开
    区间：本轮需要新计算的 prompt 范围是
    ``fill_ids[prefix_len:target_fill_len]``，因此本轮实际 extend 长度为
    ``target_fill_len - prefix_len``。它们不是 KV slot id，也不是物理页数。

    Attributes:
        req: 被决策的原始请求对象。dataclass 虽然是 frozen 的，但这里只
            冻结字段引用，``Req`` 自身仍然可变；scheduler 后续会更新它的
            ``fill_len``、状态及 KV 进度。
        result: 本轮处理结论。``ADMIT`` 表示本轮完成全部 prompt；
            ``CHUNK`` 表示只完成一段；``DEFER`` 表示当前资源不足、以后
            重试；``ABORT`` 表示请求在当前物理容量下无法运行。
        prefix_len: 本轮开始前已经具有可用 KV 的 prompt token 数。新请求
            取 radix cache 的匹配长度；续跑的 chunked 请求取自己的
            ``kv_committed_len``。该边界之前的 token 无需在本轮重算。
            “request rows full” 在计算 prefix 前就返回，因此该特殊
            ``DEFER`` 决策中该值用 0 占位，不代表真实 cache 命中长度。
        target_fill_len: 若本轮被接纳，forward 后期望到达的 prompt 绝对
            长度。``ADMIT`` 时通常等于 ``len(req.fill_ids)``；``CHUNK``
            时小于它；``DEFER`` / ``ABORT`` 时通常等于 ``prefix_len``，
            表示本轮不推进。该值会由 scheduler 写入 ``req.fill_len``，
            再由 batch 据此分配 ``[prefix_len, target_fill_len)`` 的 KV。
        reason: 不接纳时给日志和 abort 信息使用的人类可读原因。
            ``ADMIT`` / ``CHUNK`` 通常为 ``None``；``DEFER`` / ``ABORT``
            通常记录资源不足的具体原因，不参与预算计算或流程分支。

    核心关系：
        ``0 <= prefix_len <= target_fill_len <= len(req.fill_ids)``。
        唯一的占位例外是尚未计算 prefix 的 ``request rows full`` 决策。
    """

    req: Req
    result: AddReqResult
    prefix_len: int
    target_fill_len: int
    reason: str | None = None

    @property
    def is_accepted(self) -> bool:
        """请求本轮是否应该进入 EXTEND batch。"""
        return self.result in (AddReqResult.ADMIT, AddReqResult.CHUNK)

    @property
    def extend_len(self) -> int:
        """该决策要求本轮新增计算的 prompt token 数。"""
        return self.target_fill_len - self.prefix_len


@dataclass
class MemoryBudget:
    """PrefillAdder 在一次选批过程中的可变预算账本。"""

    free_tokens: int  # allocator 当前直接可用的 slot。
    evictable_tokens: int  # 可通过 cache 淘汰回收的 slot。
    protected_tokens: int  # 被活跃请求锁住、不可淘汰的 cache slot。
    decode_reserved_tokens: int  # 为 running 请求未来 decode 预留的 slot。
    remaining_prefill_tokens: int  # 本轮还允许处理多少 prompt token。
    remaining_tokens: int  # 扣除 decode 预留后仍可用于新请求的 slot。


class PrefillAdder:
    """按 FCFS 顺序为一次 EXTEND batch 选择请求。

    使用顺序：构造预算 -> ``add_requests`` 逐个决策 -> scheduler 根据结果
    真正分配 row/KV。实例只服务于一次选批，不跨 scheduler step 复用。
    """

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
        """按 chunked 优先、waiting FCFS 的顺序返回逐请求决策。"""

        decisions: list[AdmissionDecision] = []
        candidates: list[Req] = []
        if chunked_req is not None:
            candidates.append(chunked_req)
        candidates.extend(waiting_reqs)

        accepted = 0
        for req in candidates:
            # chunked_req 已经占有 row；max_new_reqs 只限制新的 waiting 请求。
            if accepted >= max_new_reqs and req is not chunked_req:
                decisions.append(
                    AdmissionDecision(req, AddReqResult.DEFER, 0, 0, "request rows full")
                )
                continue

            decision = self._decide(req, continuing=req is chunked_req)
            decisions.append(decision)
            if decision.is_accepted:
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
        """从账本扣除一个已接纳请求的本轮成本。"""
        is_last = decision.target_fill_len >= len(req.fill_ids)
        cost = self._candidate_cost(
            req, decision.prefix_len, decision.target_fill_len, is_last=is_last
        )
        extend = decision.extend_len
        self.budget.remaining_tokens = max(self.budget.remaining_tokens - cost, 0)
        self.budget.remaining_prefill_tokens = max(
            self.budget.remaining_prefill_tokens - extend, 0
        )
        self.num_accepted += 1

    def _match_len(self, req: Req) -> int:
        """只查询 radix 命中长度，不 pin cache。"""
        if self.tree_cache is None:
            return 0
        return self.tree_cache.match_prefix(req.fill_ids, pin=False).token_count

    def _round_page(self, tokens: int) -> int:
        """将 token 数向上对齐到 allocator page size。"""
        if tokens <= 0:
            return 0
        page = self.allocator.page_size
        return ((tokens + page - 1) // page) * page
