from __future__ import annotations

from abc import ABC
from collections import deque
from collections.abc import Iterable, Sequence

import numpy as np

from my_sglang.models import Req


class ReqToTokenPool:
    """请求行分配器与 ``(row, seq_pos) -> kv_slot`` 二维矩阵。"""

    def __init__(self, capacity: int, max_context_len: int):
        if capacity <= 0 or max_context_len <= 0:
            raise ValueError("ReqToTokenPool dimensions must be positive")
        self.capacity = capacity
        self.max_context_len = max_context_len
        self.req_to_token = np.full(
            (capacity, max_context_len), -1, dtype=np.int64
        )
        self._free_indices = deque(range(capacity))
        self._rid_to_idx: dict[str, int] = {}

    def alloc(self, reqs: Sequence[Req]) -> list[int] | None:
        if len(reqs) > len(self._free_indices):
            return None
        if len({req.rid for req in reqs}) != len(reqs):
            raise ValueError("duplicate rid in allocation batch")
        for req in reqs:
            if req.rid in self._rid_to_idx:
                raise ValueError(f"duplicate rid {req.rid!r}")

        indices = [self._free_indices.popleft() for _ in reqs]
        for req, idx in zip(reqs, indices, strict=True):
            self._rid_to_idx[req.rid] = idx
            self.req_to_token[idx].fill(-1)
        return indices

    def alloc_one(self, req: Req) -> int:
        indices = self.alloc([req])
        if indices is None:
            raise RuntimeError("ReqToTokenPool exhausted")
        return indices[0]

    def free(self, req: Req) -> None:
        idx = self._rid_to_idx.pop(req.rid, None)
        if idx is None:
            return
        self.req_to_token[idx].fill(-1)
        self._free_indices.append(idx)

    def write(self, req_pool_idx: int, start: int, values: Iterable[int]) -> None:
        values_array = np.fromiter(values, dtype=np.int64)
        end = start + len(values_array)
        if start < 0 or end > self.max_context_len:
            raise RuntimeError("ReqToTokenPool write exceeds max_context_len")
        self.req_to_token[req_pool_idx, start:end] = values_array

    def get(self, req_pool_idx: int, seq_pos: int) -> int | None:
        value = int(self.req_to_token[req_pool_idx, seq_pos])
        return None if value < 0 else value

    def row(self, req_pool_idx: int, end: int) -> np.ndarray:
        return self.req_to_token[req_pool_idx, :end].copy()

    def clear_range(self, req_pool_idx: int, start: int, end: int) -> None:
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
        # page 0 永远保留给 padding；可用 page 从 1 开始。
        self._free_pages = deque(range(1, self.num_pages + 1))
        self._allocated_pages: set[int] = set()

    def _page_slots(self, page: int) -> np.ndarray:
        start = page * self.page_size
        return np.arange(start, start + self.page_size, dtype=np.int64)

    def _alloc_pages(self, count: int) -> list[int] | None:
        if count > len(self._free_pages):
            return None
        pages = [self._free_pages.popleft() for _ in range(count)]
        self._allocated_pages.update(pages)
        return pages

    def alloc(self, need_size: int) -> np.ndarray | None:
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
        if not (len(prefix_lens) == len(seq_lens) == len(last_locs)):
            raise ValueError("extend allocation inputs must have equal lengths")

        tails: list[np.ndarray] = []
        remaining: list[int] = []
        total_new_pages = 0
        for prefix_len, seq_len, last_loc in zip(
            prefix_lens, seq_lens, last_locs, strict=True
        ):
            need = int(seq_len) - int(prefix_len)
            if need < 0:
                raise ValueError("seq_len must be >= prefix_len")
            tail = self._tail_slots(int(last_loc), need)
            rem = need - len(tail)
            tails.append(tail)
            remaining.append(rem)
            total_new_pages += (rem + self.page_size - 1) // self.page_size

        pages = self._alloc_pages(total_new_pages)
        if pages is None:
            return None

        page_cursor = 0
        outputs: list[np.ndarray] = []
        for tail, rem in zip(tails, remaining, strict=True):
            num_pages = (rem + self.page_size - 1) // self.page_size
            req_pages = pages[page_cursor : page_cursor + num_pages]
            page_cursor += num_pages
            new_slots = (
                np.concatenate([self._page_slots(page) for page in req_pages])[:rem]
                if req_pages
                else np.empty((0,), dtype=np.int64)
            )
            outputs.append(np.concatenate((tail, new_slots)))
        return tuple(outputs)

    def alloc_decode(
        self,
        seq_lens: Sequence[int],
        last_locs: Sequence[int],
    ) -> np.ndarray | None:
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
        need = max(seq_len - prefix_len, 0)
        if need == 0:
            return 0
        tail = len(self._tail_slots(last_loc, need)) if last_loc >= 0 else 0
        return (need - tail + self.page_size - 1) // self.page_size

    def required_pages_for_decode(
        self, seq_lens: Sequence[int], last_locs: Sequence[int]
    ) -> int:
        return sum(
            self.required_pages_for_extend(int(seq_len) - 1, int(seq_len), int(last))
            for seq_len, last in zip(seq_lens, last_locs, strict=True)
        )

    def free(self, slots: Sequence[int] | np.ndarray) -> None:
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
        return slot // self.page_size in self._allocated_pages

    def assert_consistent(self) -> None:
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
    if page_size == 1:
        return TokenToKVPoolAllocator(capacity)
    return PagedTokenToKVPoolAllocator(capacity, page_size)
