"""请求行映射和物理 KV page 分配。

先记住一条链路：``Req.req_pool_idx`` 找到请求行，行内的 ``seq_pos`` 保存
物理 ``kv_slot``，而连续的一组 slot 属于同一个 page。本模块只管理整数索引，
不保存模型真实的 K/V 张量。
"""

from __future__ import annotations

from abc import ABC
from collections import deque
from collections.abc import Iterable, Sequence

import numpy as np

from my_sglang.models import Req


class ReqToTokenPool:
    """分配请求行，并保存 ``(request_row, seq_pos) -> kv_slot`` 映射。"""

    def __init__(self, capacity: int, max_context_len: int):
        if capacity <= 0 or max_context_len <= 0:
            raise ValueError("ReqToTokenPool dimensions must be positive")
        self.capacity = capacity
        self.max_context_len = max_context_len
        # -1 表示该逻辑 token 位置还没有绑定物理 KV slot。
        self.req_to_token = np.full(
            (capacity, max_context_len), -1, dtype=np.int64
        )
        # 空闲行按 FIFO 复用，方便测试稳定观察行号。
        self._free_indices = deque(range(capacity))
        # rid 用来防止同一个请求重复占用两行。
        self._rid_to_idx: dict[str, int] = {}

    def alloc(self, reqs: Sequence[Req]) -> list[int] | None:
        """为一组请求分配行；容量不足返回 ``None``，不做部分分配。"""
        if len(reqs) > len(self._free_indices):
            return None
        if len({req.rid for req in reqs}) != len(reqs):
            raise ValueError("duplicate rid in allocation batch")
        for req in reqs:
            if req.rid in self._rid_to_idx:
                raise ValueError(f"duplicate rid {req.rid!r}")

        allocated_rows = [self._free_indices.popleft() for _ in reqs]
        for req, row_index in zip(reqs, allocated_rows, strict=True):
            self._rid_to_idx[req.rid] = row_index
            self.req_to_token[row_index].fill(-1)
        return allocated_rows

    def alloc_one(self, req: Req) -> int:
        """为单个请求分配一行；容量不足直接抛错。"""
        indices = self.alloc([req])
        if indices is None:
            raise RuntimeError("ReqToTokenPool exhausted")
        return indices[0]

    def free(self, req: Req) -> None:
        """释放请求行并清空其中的全部 slot 映射。"""
        row_index = self._rid_to_idx.pop(req.rid, None)
        if row_index is None:
            return
        self.req_to_token[row_index].fill(-1)
        self._free_indices.append(row_index)

    def write(self, req_pool_idx: int, start: int, values: Iterable[int]) -> None:
        """从 ``start`` 开始写入一段连续的 KV slot 映射。"""
        values_array = np.fromiter(values, dtype=np.int64)
        end = start + len(values_array)
        if start < 0 or end > self.max_context_len:
            raise RuntimeError("ReqToTokenPool write exceeds max_context_len")
        self.req_to_token[req_pool_idx, start:end] = values_array

    def get(self, req_pool_idx: int, seq_pos: int) -> int | None:
        """读取一个逻辑位置对应的 KV slot；未映射返回 ``None``。"""
        value = int(self.req_to_token[req_pool_idx, seq_pos])
        return None if value < 0 else value

    def row(self, req_pool_idx: int, end: int) -> np.ndarray:
        """复制请求行的 ``[0, end)`` 区间，避免调用者误改原表。"""
        return self.req_to_token[req_pool_idx, :end].copy()

    def clear_range(self, req_pool_idx: int, start: int, end: int) -> None:
        """将请求行的 ``[start, end)`` 恢复为未映射状态。"""
        self.req_to_token[req_pool_idx, start:end] = -1

    def get_req_index(self, rid: str) -> int | None:
        return self._rid_to_idx.get(rid)

    @property
    def active_count(self) -> int:
        return len(self._rid_to_idx)

    @property
    def available_size(self) -> int:
        return len(self._free_indices)

    @property
    def mapped_size(self) -> int:
        return int(np.count_nonzero(self.req_to_token >= 0))

    @property
    def size(self) -> int:
        """Compatibility/readability alias for the number of mapped positions."""
        return self.mapped_size

    def assert_consistent(self) -> None:
        """检查每一行恰好处于 active 或 free 两种状态之一。"""
        active_indices = set(self._rid_to_idx.values())
        if len(active_indices) != len(self._rid_to_idx):
            raise AssertionError("multiple requests share a request row")
        if active_indices & set(self._free_indices):
            raise AssertionError("request row is both active and free")
        if len(active_indices) + len(self._free_indices) != self.capacity:
            raise AssertionError("request row accounting mismatch")


class BaseTokenToKVPoolAllocator(ABC):
    """只管理 KV slot/page 索引，不保存真实 K/V 张量。"""

    def __init__(self, capacity: int, page_size: int):
        if capacity <= 0 or page_size <= 0:
            raise ValueError("KV allocator dimensions must be positive")
        if capacity % page_size != 0:
            raise ValueError("max_total_tokens must be divisible by page_size")
        self.capacity = capacity
        self.page_size = page_size
        self.num_pages = capacity // page_size
        # page 0 永远保留给 padding；真正可用 page 编号从 1 开始。
        self._free_pages = deque(range(1, self.num_pages + 1))
        self._allocated_pages: set[int] = set()

    def _page_slots(self, page: int) -> np.ndarray:
        """展开一页中的全部连续 slot。"""
        start = page * self.page_size
        return np.arange(start, start + self.page_size, dtype=np.int64)

    def _alloc_pages(self, count: int) -> list[int] | None:
        """原子地申请整页；页数不足时不改变任何状态。"""
        if count > len(self._free_pages):
            return None
        pages = [self._free_pages.popleft() for _ in range(count)]
        self._allocated_pages.update(pages)
        return pages

    def alloc(self, need_size: int) -> np.ndarray | None:
        """按页申请至少 ``need_size`` 个 slot，只返回调用方需要的部分。"""
        if need_size < 0:
            raise ValueError("need_size must be non-negative")
        if need_size == 0:
            return np.empty((0,), dtype=np.int64)
        num_pages = (need_size + self.page_size - 1) // self.page_size
        pages = self._alloc_pages(num_pages)
        if pages is None:
            return None
        slots = np.concatenate([self._page_slots(page) for page in pages])
        return slots[:need_size]

    def _tail_slots(self, last_loc: int, need_size: int) -> np.ndarray:
        """优先返回最后一页尚未使用的连续 slot，减少新 page 分配。"""
        if last_loc < 0 or need_size <= 0:
            return np.empty((0,), dtype=np.int64)
        page = last_loc // self.page_size
        offset = last_loc % self.page_size
        if page == 0 or page not in self._allocated_pages:
            raise RuntimeError(f"last_loc {last_loc} is not in an allocated page")
        take = min(self.page_size - offset - 1, need_size)
        return np.arange(last_loc + 1, last_loc + 1 + take, dtype=np.int64)

    def alloc_extend(
        self,
        prefix_lens: Sequence[int],
        seq_lens: Sequence[int],
        last_locs: Sequence[int],
    ) -> tuple[np.ndarray, ...] | None:
        """为多个请求批量分配 EXTEND slot。

        每个请求先复用自己的尾页空间，再统一申请新页。整个操作要么全部
        成功，要么返回 ``None``，不会只给部分请求分配资源。
        """
        if not (len(prefix_lens) == len(seq_lens) == len(last_locs)):
            raise ValueError("extend allocation inputs must have equal lengths")

        reusable_tails: list[np.ndarray] = []
        remaining_sizes: list[int] = []
        total_new_pages = 0
        for prefix_len, seq_len, last_loc in zip(
            prefix_lens, seq_lens, last_locs, strict=True
        ):
            required_slots = int(seq_len) - int(prefix_len)
            if required_slots < 0:
                raise ValueError("seq_len must be >= prefix_len")
            reusable_tail = self._tail_slots(int(last_loc), required_slots)
            remaining_size = required_slots - len(reusable_tail)
            reusable_tails.append(reusable_tail)
            remaining_sizes.append(remaining_size)
            total_new_pages += (
                remaining_size + self.page_size - 1
            ) // self.page_size

        pages = self._alloc_pages(total_new_pages)
        if pages is None:
            return None

        next_page_index = 0
        allocated_slots_by_req: list[np.ndarray] = []
        for reusable_tail, remaining_size in zip(
            reusable_tails, remaining_sizes, strict=True
        ):
            num_pages = (
                remaining_size + self.page_size - 1
            ) // self.page_size
            request_pages = pages[next_page_index : next_page_index + num_pages]
            next_page_index += num_pages
            new_slots = (
                np.concatenate(
                    [self._page_slots(page) for page in request_pages]
                )[:remaining_size]
                if request_pages
                else np.empty((0,), dtype=np.int64)
            )
            allocated_slots_by_req.append(
                np.concatenate((reusable_tail, new_slots))
            )
        return tuple(allocated_slots_by_req)

    def alloc_decode(
        self,
        seq_lens: Sequence[int],
        last_locs: Sequence[int],
    ) -> np.ndarray | None:
        """为每个 decode 请求分配恰好一个新 token 的 slot。"""
        outputs = self.alloc_extend(
            prefix_lens=[int(seq_len) - 1 for seq_len in seq_lens],
            seq_lens=seq_lens,
            last_locs=last_locs,
        )
        if outputs is None:
            return None
        return np.asarray([int(slots[0]) for slots in outputs], dtype=np.int64)

    def required_pages_for_extend(
        self, prefix_len: int, seq_len: int, last_loc: int
    ) -> int:
        """只计算 EXTEND 还需多少新 page，不改变 allocator 状态。"""
        need = max(seq_len - prefix_len, 0)
        if need == 0:
            return 0
        tail = len(self._tail_slots(last_loc, need)) if last_loc >= 0 else 0
        return (need - tail + self.page_size - 1) // self.page_size

    def required_pages_for_decode(
        self, seq_lens: Sequence[int], last_locs: Sequence[int]
    ) -> int:
        """计算整个 decode batch 还需多少新 page。"""
        return sum(
            self.required_pages_for_extend(int(seq_len) - 1, int(seq_len), int(last))
            for seq_len, last in zip(seq_lens, last_locs, strict=True)
        )

    def free(self, slots: Sequence[int] | np.ndarray) -> None:
        """释放 ``slots`` 所在的整页；重复 slot/page 会自动去重。"""
        array = np.asarray(slots, dtype=np.int64)
        if array.size == 0:
            return
        pages = sorted({int(slot) // self.page_size for slot in array if slot >= 0})
        for page in pages:
            if page == 0:
                continue
            if page in self._allocated_pages:
                self._allocated_pages.remove(page)
                self._free_pages.append(page)

    def free_unshared_pages(
        self,
        candidate_slots: Sequence[int] | np.ndarray,
        protected_slots: Sequence[int] | np.ndarray,
    ) -> None:
        """释放候选 slot 中不与保护区域共页的 page。"""
        protected_pages = {
            int(slot) // self.page_size
            for slot in np.asarray(protected_slots, dtype=np.int64)
            if slot >= 0
        }
        releasable = [
            int(slot)
            for slot in np.asarray(candidate_slots, dtype=np.int64)
            if slot >= 0 and int(slot) // self.page_size not in protected_pages
        ]
        self.free(releasable)

    @property
    def available_size(self) -> int:
        return len(self._free_pages) * self.page_size

    @property
    def allocated_size(self) -> int:
        return len(self._allocated_pages) * self.page_size

    @property
    def active_count(self) -> int:
        return self.allocated_size

    def owns_slot(self, slot: int) -> bool:
        """判断一个 slot 所在 page 当前是否由 allocator 持有。"""
        return slot // self.page_size in self._allocated_pages

    def assert_consistent(self) -> None:
        """检查 page 不会同时处于 free 和 allocated。"""
        free = set(self._free_pages)
        if free & self._allocated_pages:
            raise AssertionError("KV page is both free and allocated")
        if len(free) + len(self._allocated_pages) != self.num_pages:
            raise AssertionError("KV page accounting mismatch")
        if 0 in free or 0 in self._allocated_pages:
            raise AssertionError("padding page 0 must never be allocated")


class TokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    def __init__(self, capacity: int):
        super().__init__(capacity, page_size=1)


class PagedTokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    def __init__(self, capacity: int, page_size: int):
        if page_size <= 1:
            raise ValueError("PagedTokenToKVPoolAllocator requires page_size > 1")
        super().__init__(capacity, page_size=page_size)


def build_token_allocator(
    capacity: int, page_size: int
) -> BaseTokenToKVPoolAllocator:
    """按 page size 创建逐 token 或分页 allocator。"""
    if page_size == 1:
        return TokenToKVPoolAllocator(capacity)
    return PagedTokenToKVPoolAllocator(capacity, page_size)
