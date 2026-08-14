from __future__ import annotations

import numpy as np
import pytest

from my_sglang.models import Req, SamplingParams
from my_sglang.pools import (
    PagedTokenToKVPoolAllocator,
    ReqToTokenPool,
    TokenToKVPoolAllocator,
)


def req(rid: str) -> Req:
    return Req(rid, [1], SamplingParams(max_new_tokens=1))


def test_req_to_token_pool_allocates_fixed_numpy_rows_and_reuses_them():
    pool = ReqToTokenPool(size=2, max_context_len=4)
    a, b = req("a"), req("b")
    rows = pool.alloc([a, b])

    assert rows is not None
    assert rows == [0, 1]
    assert pool.req_to_token.dtype == np.int64
    pool.write(rows[0], 0, [7, 8])
    assert tuple(pool.row(rows[0], 3)) == (7, 8, -1)
    pool.free(a)
    assert pool.alloc_one(req("c")) == 0
    pool.assert_consistent()


def test_token_allocator_reserves_slot_zero():
    allocator = TokenToKVPoolAllocator(3)
    slots = allocator.alloc(2)
    assert slots is not None
    assert tuple(slots) == (1, 2)
    assert allocator.available_size == 1
    allocator.free([1, 2])
    assert allocator.available_size == 3
    allocator.assert_consistent()


def test_paged_allocator_reuses_tail_before_allocating_next_page():
    allocator = PagedTokenToKVPoolAllocator(size=6, page_size=2)
    prompt = allocator.alloc_extend([0], [3], [-1])
    assert prompt is not None
    assert tuple(prompt[0]) == (2, 3, 4)
    assert allocator.allocated_size == 4

    decode = allocator.alloc_decode([4], [4])
    assert decode is not None
    assert tuple(decode) == (5,)
    assert allocator.allocated_size == 4

    next_decode = allocator.alloc_decode([5], [5])
    assert next_decode is not None
    assert tuple(next_decode) == (6,)
    assert allocator.allocated_size == 6
    allocator.free([2, 4, 6])
    assert allocator.available_size == 6


def test_paged_allocator_validates_page_geometry():
    with pytest.raises(ValueError, match="divisible"):
        PagedTokenToKVPoolAllocator(size=5, page_size=2)
