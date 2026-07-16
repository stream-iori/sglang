from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from my_sglang.models import BatchForward, ForwardMode, Req, RequestStatus
from my_sglang.pools import BaseTokenToKVPoolAllocator, ReqToTokenPool

if TYPE_CHECKING:
    from my_sglang.radix_cache import MiniRadixCache


@dataclass
class MiniScheduleBatch:
    """调度器内部可变 batch；BatchForward 是它对 runner 的不可变视图。"""

    reqs: list[Req]
    forward_mode: ForwardMode
    req_to_token_pool: ReqToTokenPool
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator
    tree_cache: MiniRadixCache | None
    chunked_req: Req | None = None
    first_extend_by_req: tuple[bool, ...] = ()
    input_ids_by_req: tuple[tuple[int, ...], ...] = ()
    req_pool_indices: np.ndarray = field(
        default_factory=lambda: np.empty((0,), dtype=np.int64)
    )
    out_cache_locs_by_req: tuple[np.ndarray, ...] = ()
    seq_lens: np.ndarray = field(
        default_factory=lambda: np.empty((0,), dtype=np.int64)
    )
    prefix_lens: np.ndarray = field(
        default_factory=lambda: np.empty((0,), dtype=np.int64)
    )
    extend_lens: np.ndarray = field(
        default_factory=lambda: np.empty((0,), dtype=np.int64)
    )
    chunk_starts: np.ndarray = field(
        default_factory=lambda: np.empty((0,), dtype=np.int64)
    )
    is_last_prefill_chunk: tuple[bool, ...] = ()

    @classmethod
    def empty(
        cls,
        req_to_token_pool: ReqToTokenPool,
        allocator: BaseTokenToKVPoolAllocator,
        tree_cache: MiniRadixCache | None,
    ) -> MiniScheduleBatch:
        return cls([], ForwardMode.DECODE, req_to_token_pool, allocator, tree_cache)

    def is_empty(self) -> bool:
        return not self.reqs

    def batch_size(self) -> int:
        return len(self.reqs)

    def prepare_for_extend(self) -> None:
        # 从 Req 的 allocated 边界到 fill_len 统一分配；allocator 会先利用
        # 分页尾部，再申请新 page。本方法只推进 allocated。
        if self.forward_mode is not ForwardMode.EXTEND:
            raise RuntimeError("prepare_for_extend requires EXTEND mode")

        starts = [req.kv_allocated_len for req in self.reqs]
        targets = [req.fill_len for req in self.reqs]
        last_locs = [self._last_loc(req, start) for req, start in zip(self.reqs, starts)]
        allocated = self.token_to_kv_pool_allocator.alloc_extend(
            starts, targets, last_locs
        )
        if allocated is None:
            raise RuntimeError("KV pool exhausted while preparing extend batch")

        inputs: list[tuple[int, ...]] = []
        for req, start, target, slots in zip(
            self.reqs, starts, targets, allocated, strict=True
        ):
            req_pool_idx = self._require_req_pool_idx(req)
            self.req_to_token_pool.write(req_pool_idx, start, slots)
            inputs.append(tuple(req.fill_ids[start:target]))
            req.kv_allocated_len = target
            req.extend_input_len = target - start

        self.req_pool_indices = np.asarray(
            [self._require_req_pool_idx(req) for req in self.reqs], dtype=np.int64
        )
        self.out_cache_locs_by_req = allocated
        self.input_ids_by_req = tuple(inputs)
        self.seq_lens = np.asarray(targets, dtype=np.int64)
        self.prefix_lens = np.asarray(starts, dtype=np.int64)
        self.extend_lens = self.seq_lens - self.prefix_lens
        self.chunk_starts = self.prefix_lens.copy()
        self.is_last_prefill_chunk = tuple(
            target >= len(req.fill_ids)
            for req, target in zip(self.reqs, targets, strict=True)
        )

    def prepare_for_decode(self) -> None:
        # decode 把“上一个生成 token”作为本轮输入，每个请求增加
        # 一个序列位置；是否真正 committed 仍由 forward 成功后决定。
        if self.forward_mode is not ForwardMode.DECODE:
            raise RuntimeError("prepare_for_decode requires DECODE mode")
        starts = [req.kv_allocated_len for req in self.reqs]
        targets = [start + 1 for start in starts]
        last_locs = [self._last_loc(req, start) for req, start in zip(self.reqs, starts)]
        slots = self.token_to_kv_pool_allocator.alloc_decode(targets, last_locs)
        if slots is None:
            raise RuntimeError("KV pool exhausted while preparing decode batch")

        for req, start, slot in zip(self.reqs, starts, slots, strict=True):
            req_pool_idx = self._require_req_pool_idx(req)
            self.req_to_token_pool.write(req_pool_idx, start, [int(slot)])
            req.kv_allocated_len = start + 1

        self.req_pool_indices = np.asarray(
            [self._require_req_pool_idx(req) for req in self.reqs], dtype=np.int64
        )
        self.out_cache_locs_by_req = tuple(
            np.asarray([slot], dtype=np.int64) for slot in slots
        )
        self.input_ids_by_req = tuple((req.last_token_id,) for req in self.reqs)
        self.seq_lens = np.asarray(targets, dtype=np.int64)
        self.prefix_lens = np.asarray(starts, dtype=np.int64)
        self.extend_lens = np.ones((len(self.reqs),), dtype=np.int64)
        self.chunk_starts = self.prefix_lens.copy()
        self.is_last_prefill_chunk = ()

    def commit_allocated(self) -> None:
        # forward 成功后才能推进 committed；overlap launch 后暂时不能调用。
        for req in self.reqs:
            if req.kv_committed_len > req.kv_allocated_len:
                raise AssertionError("committed KV exceeds allocated KV")
            req.kv_committed_len = req.kv_allocated_len

    def rollback_uncommitted(self) -> None:
        # 只释放完全位于 committed 边界之后的 page；共享尾页必须保留。
        for req in self.reqs:
            if req.req_pool_idx is None:
                continue
            committed = self.req_to_token_pool.row(
                req.req_pool_idx, req.kv_committed_len
            )
            allocated = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, req.kv_committed_len : req.kv_allocated_len
            ].copy()
            self.token_to_kv_pool_allocator.free_unshared_pages(
                allocated, committed
            )
            self.req_to_token_pool.clear_range(
                req.req_pool_idx, req.kv_committed_len, req.kv_allocated_len
            )
            req.kv_allocated_len = req.kv_committed_len

    def filter_batch(self, exclude: set[Req] | None = None) -> None:
        exclude = exclude or set()
        self.reqs = [
            req
            for req in self.reqs
            if req not in exclude and req.status is not RequestStatus.FINISHED
        ]

    def merge_batch(self, other: MiniScheduleBatch) -> None:
        existing = set(self.reqs)
        self.reqs.extend(req for req in other.reqs if req not in existing)

    def to_forward_batch(self) -> BatchForward:
        return BatchForward(
            mode=self.forward_mode,
            reqs=tuple(self.reqs),
            input_ids_by_req=self.input_ids_by_req,
            req_pool_indices=tuple(int(x) for x in self.req_pool_indices),
            out_cache_locs=tuple(
                tuple(int(x) for x in slots) for slots in self.out_cache_locs_by_req
            ),
            seq_lens=tuple(int(x) for x in self.seq_lens),
            prefix_slot_ids_by_req=tuple(
                tuple(int(x) for x in req.prefix_indices) for req in self.reqs
            ),
            extend_lens=tuple(int(x) for x in self.extend_lens),
            chunk_starts_by_req=tuple(int(x) for x in self.chunk_starts),
            is_last_prefill_chunk_by_req=self.is_last_prefill_chunk,
        )

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
