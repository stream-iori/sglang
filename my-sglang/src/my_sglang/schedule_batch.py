"""一次 forward batch 的准备、提交与回滚。

``MiniScheduleBatch`` 是可变的调度工作单：prepare 阶段预分配 slot，runner
成功后 commit，失败则 rollback。交给 runner 的 ``ForwardBatch`` 只是快照。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from my_sglang.models import (
    ForwardBatch,
    ForwardMode,
    Range,
    Req,
    RequestStatus,
)
from my_sglang.pools import BaseTokenToKVPoolAllocator, ReqToTokenPool

if TYPE_CHECKING:
    from my_sglang.radix_cache import MiniRadixCache


@dataclass
class MiniScheduleBatch:
    """调度器内部可变 batch；ForwardBatch 是它对 runner 的不可变视图。"""

    # 本轮参与 forward 的请求，顺序与下面所有按请求排列的字段一致。
    reqs: list[Req]
    # 本轮执行模式：EXTEND（prefill）或 DECODE。
    forward_mode: ForwardMode
    # 请求逻辑位置到物理 KV slot 的映射池。
    req_to_token_pool: ReqToTokenPool
    # 负责申请和释放物理 KV slot 的分配器。
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator
    # 可复用 prompt KV 的 radix cache；未启用时为 None。
    tree_cache: MiniRadixCache | None
    # 当前正在分块 prefill 的请求；没有时为 None。
    chunked_req: Req | None = None
    # 每个请求是否首次进入 EXTEND，用于选择 prefill 或 extend runner 接口。
    first_extend_by_req: tuple[bool, ...] = ()
    # 与标准 ScheduleBatch 一样保存展平 input ids。
    input_ids: tuple[int, ...] = ()
    # 每个请求在 ReqToTokenPool 中对应的行号。
    req_pool_indices: np.ndarray = field(
        default_factory=lambda: np.empty((0,), dtype=np.int64)
    )
    # 与标准 ScheduleBatch 一样保存展平 output KV locations。
    out_cache_loc: np.ndarray = field(
        default_factory=lambda: np.empty((0,), dtype=np.int64)
    )
    # 每个请求完成本轮后达到的总序列长度。
    seq_lens: np.ndarray = field(
        default_factory=lambda: np.empty((0,), dtype=np.int64)
    )
    # 每个请求在本轮开始前已经拥有 KV 的 token 长度。
    prefix_lens: np.ndarray = field(
        default_factory=lambda: np.empty((0,), dtype=np.int64)
    )
    # 每个请求本轮新增计算的 token 数，即 seq_lens - prefix_lens。
    extend_lens: np.ndarray = field(
        default_factory=lambda: np.empty((0,), dtype=np.int64)
    )

    @classmethod
    def empty(
        cls,
        req_to_token_pool: ReqToTokenPool,
        allocator: BaseTokenToKVPoolAllocator,
        tree_cache: MiniRadixCache | None,
    ) -> MiniScheduleBatch:
        """创建一个没有请求的占位 batch。"""
        return cls([], ForwardMode.DECODE, req_to_token_pool, allocator, tree_cache)

    def is_empty(self) -> bool:
        """当前 batch 是否没有任何请求。"""
        return not self.reqs

    def batch_size(self) -> int:
        """返回当前 batch 的请求数量。"""
        return len(self.reqs)

    def prepare_for_extend(self) -> None:
        """为 EXTEND 输入预分配 KV slot 并生成 runner 所需元数据。"""

        if self.forward_mode is not ForwardMode.EXTEND:
            raise RuntimeError("prepare_for_extend requires EXTEND mode")

        # 阶段 1：确定每个请求本轮的逻辑区间 [start, target)。
        ranges = [self._require_extend_range(req) for req in self.reqs]
        start_lens = [item.start for item in ranges]
        target_lens = [item.end for item in ranges]
        last_slot_ids = [
            self._last_loc(req, start_len)
            for req, start_len in zip(self.reqs, start_lens, strict=True)
        ]

        # 阶段 2：批量申请物理 slot。allocator 会先复用请求自己的尾页。
        new_slots_by_req = self.token_to_kv_pool_allocator.alloc_extend(
            start_lens, target_lens, last_slot_ids
        )
        if new_slots_by_req is None:
            raise RuntimeError("KV pool exhausted while preparing extend batch")

        # 阶段 3：写入 row -> slot 映射，并推进 allocated 边界。
        input_ids_by_req: list[tuple[int, ...]] = []
        for req, start_len, target_len, new_slots in zip(
            self.reqs,
            start_lens,
            target_lens,
            new_slots_by_req,
            strict=True,
        ):
            req_pool_idx = self._require_req_pool_idx(req)
            self.req_to_token_pool.write(req_pool_idx, start_len, new_slots)
            input_ids_by_req.append(tuple(req.get_fill_ids()[start_len:target_len]))
            req.kv.kv_allocated_len = target_len

        # 阶段 4：保存与 reqs 按下标对齐的 runner 元数据。
        self.req_pool_indices = np.asarray(
            [self._require_req_pool_idx(req) for req in self.reqs], dtype=np.int64
        )
        self.out_cache_loc = np.concatenate(new_slots_by_req)
        self.input_ids = tuple(
            token for request_tokens in input_ids_by_req for token in request_tokens
        )
        self.seq_lens = np.asarray(target_lens, dtype=np.int64)
        self.prefix_lens = np.asarray(start_lens, dtype=np.int64)
        self.extend_lens = self.seq_lens - self.prefix_lens

    def prepare_for_decode(self) -> None:
        """为每个 DECODE 请求预分配一个 slot 并准备输入 token。"""

        if self.forward_mode is not ForwardMode.DECODE:
            raise RuntimeError("prepare_for_decode requires DECODE mode")
        # 每个 decode 请求只前进一个逻辑位置。
        start_lens = [req.kv.kv_allocated_len for req in self.reqs]
        target_lens = [start_len + 1 for start_len in start_lens]
        last_slot_ids = [
            self._last_loc(req, start_len)
            for req, start_len in zip(self.reqs, start_lens, strict=True)
        ]
        new_slots = self.token_to_kv_pool_allocator.alloc_decode(
            target_lens, last_slot_ids
        )
        if new_slots is None:
            raise RuntimeError("KV pool exhausted while preparing decode batch")

        for req, start_len, new_slot in zip(
            self.reqs, start_lens, new_slots, strict=True
        ):
            req_pool_idx = self._require_req_pool_idx(req)
            self.req_to_token_pool.write(req_pool_idx, start_len, [int(new_slot)])
            req.kv.kv_allocated_len = start_len + 1
            req.extend_range = Range(start_len, start_len + 1)

        self.req_pool_indices = np.asarray(
            [self._require_req_pool_idx(req) for req in self.reqs], dtype=np.int64
        )
        self.out_cache_loc = np.asarray(new_slots, dtype=np.int64)
        # overlap 路径在 Fake CUDA forward stream 中从 FutureMap gather；同步路径
        # 仍可直接使用请求最后一个已提交 token。
        self.input_ids = tuple(req.last_token_id for req in self.reqs)
        self.seq_lens = np.asarray(target_lens, dtype=np.int64)
        self.prefix_lens = np.asarray(start_lens, dtype=np.int64)
        self.extend_lens = np.ones((len(self.reqs),), dtype=np.int64)

    def commit_allocated(self) -> None:
        """模型成功后，将每个请求的 committed 边界追平 allocated。"""
        # 生产 pipeline 在 launch 成功后推进 KV 逻辑水位；延迟的是 CPU
        # token、finish 和资源释放处理。
        for req in self.reqs:
            if req.kv_committed_len > req.kv.kv_allocated_len:
                raise AssertionError("committed KV exceeds allocated KV")
            req.kv_committed_len = req.kv.kv_allocated_len

    def rollback_uncommitted(self) -> None:
        """模型失败后清除未提交映射，并释放不与 committed 区域共页的 page。"""
        for req in self.reqs:
            if req.req_pool_idx is None:
                continue
            committed_slots = self.req_to_token_pool.row(
                req.req_pool_idx, req.kv_committed_len
            )
            uncommitted_slots = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, req.kv_committed_len : req.kv.kv_allocated_len
            ].copy()
            self.token_to_kv_pool_allocator.free_unshared_pages(
                uncommitted_slots, committed_slots
            )
            self.req_to_token_pool.clear_range(
                req.req_pool_idx, req.kv_committed_len, req.kv.kv_allocated_len
            )
            req.kv.kv_allocated_len = req.kv_committed_len

    def filter_batch(self, exclude: set[Req] | None = None) -> None:
        """移除已结束请求，以及调用方明确排除的请求。"""
        exclude = exclude or set()
        self.reqs = [
            req
            for req in self.reqs
            if req not in exclude and req.status is not RequestStatus.FINISHED
        ]

    def merge_batch(self, other: MiniScheduleBatch) -> None:
        """按对象身份追加另一个 batch 中尚未存在的请求。"""
        existing = set(self.reqs)
        for req in other.reqs:
            if req not in existing:
                self.reqs.append(req)
                existing.add(req)

    def to_forward_batch(self) -> ForwardBatch:
        """将可变调度 batch 转成 runner 使用的不可变快照。"""

        prefix_slots_by_req: list[tuple[int, ...]] = []
        for req in self.reqs:
            prefix_slots_by_req.append(
                tuple(int(slot) for slot in req.prefix_indices)
            )

        return ForwardBatch(
            forward_mode=self.forward_mode,
            reqs=tuple(self.reqs),
            input_ids=self.input_ids,
            req_pool_indices=tuple(int(x) for x in self.req_pool_indices),
            out_cache_loc=tuple(int(slot) for slot in self.out_cache_loc),
            seq_lens=tuple(int(x) for x in self.seq_lens),
            prefix_indices_by_req=tuple(prefix_slots_by_req),
            extend_seq_lens=tuple(int(x) for x in self.extend_lens),
            extend_range_starts=tuple(
                self._require_extend_range(req).start for req in self.reqs
            ),
            contains_last_prefill_chunk=all(
                req.extend_range is not None
                and req.extend_range.end >= len(req.get_fill_ids())
                for req in self.reqs
            ),
        )

    @property
    def input_ids_by_req(self) -> tuple[tuple[int, ...], ...]:
        return self._split_flat(self.input_ids)

    @property
    def out_cache_loc_by_req(self) -> tuple[np.ndarray, ...]:
        return tuple(
            np.asarray(group, dtype=np.int64)
            for group in self._split_flat(tuple(int(x) for x in self.out_cache_loc))
        )

    def _split_flat(self, values: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
        groups: list[tuple[int, ...]] = []
        offset = 0
        for length in self.extend_lens:
            size = int(length)
            groups.append(values[offset : offset + size])
            offset += size
        if offset != len(values):
            raise AssertionError("flat ScheduleBatch field does not match extend_lens")
        return tuple(groups)

    def _last_loc(self, req: Req, length: int) -> int:
        if length == 0:
            return -1
        return int(self.req_to_token_pool.req_to_token[
            self._require_req_pool_idx(req), length - 1
        ])

    @staticmethod
    def _require_req_pool_idx(req: Req) -> int:
        if req.req_pool_idx is None:
            raise RuntimeError(f"request {req.rid} has no req_pool_idx")
        return req.req_pool_idx

    @staticmethod
    def _require_extend_range(req: Req) -> Range:
        if req.extend_range is None:
            raise RuntimeError(f"request {req.rid} has no extend_range")
        return req.extend_range
