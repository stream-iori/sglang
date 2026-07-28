from __future__ import annotations

from my_sglang.models import ForwardMode, Req, SamplingParams
from my_sglang.pools import ReqToTokenPool, TokenToKVPoolAllocator
from my_sglang.radix_cache import MiniRadixCache
from my_sglang.schedule_batch import MiniScheduleBatch
from my_sglang.schedule_policy import AddReqResult, PrefillAdder


def req(rid: str, ids: list[int], max_new_tokens: int = 2) -> Req:
    return Req(rid, ids, SamplingParams(max_new_tokens=max_new_tokens))


def empty_batch(allocator, cache=None):
    return MiniScheduleBatch.empty(ReqToTokenPool(4, 16), allocator, cache)


def test_prefill_adder_stops_at_first_fcfs_budget_defer():
    allocator = TokenToKVPoolAllocator(12)
    adder = PrefillAdder(
        allocator,
        None,
        empty_batch(allocator),
        max_prefill_tokens=2,
        chunked_prefill_size=None,
        new_token_ratio=0,
    )

    decisions = adder.add_requests(
        [req("a", [1, 2]), req("b", [3, 4])], None, max_new_reqs=4
    )

    assert [decision.result for decision in decisions] == [
        AddReqResult.ADMIT,
        AddReqResult.DEFER,
    ]
    assert adder.budget.remaining_prefill_tokens == 0


def test_cached_prefix_reduces_extend_length_and_cache_is_evictable_budget():
    allocator = TokenToKVPoolAllocator(4)
    cached_slots = allocator.alloc(2)
    assert cached_slots is not None
    cache = MiniRadixCache()
    cache.insert([1, 2], cached_slots)
    adder = PrefillAdder(
        allocator,
        cache,
        empty_batch(allocator, cache),
        max_prefill_tokens=4,
        chunked_prefill_size=None,
        new_token_ratio=0,
    )

    decision = adder.add_requests([req("reuse", [1, 2, 3])], None, 1)[0]

    assert decision.result is AddReqResult.ADMIT
    assert decision.prefix_len == 2
    assert decision.target_fill_len == 3
    assert adder.budget.evictable_tokens == 2


def test_first_chunk_uses_physical_capacity_without_permanent_defer():
    allocator = TokenToKVPoolAllocator(2)
    adder = PrefillAdder(
        allocator,
        None,
        empty_batch(allocator),
        max_prefill_tokens=2,
        chunked_prefill_size=2,
        new_token_ratio=0.5,
    )

    decision = adder.add_requests([req("large", [1, 2, 3, 4])], None, 1)[0]

    assert decision.result is AddReqResult.CHUNK
    assert decision.target_fill_len == 2


def test_decode_reserve_uses_ratio_then_full_budget_after_retraction():
    allocator = TokenToKVPoolAllocator(8)
    running = empty_batch(allocator)
    active = req("active", [1], max_new_tokens=3)
    running.reqs = [active]
    running.forward_mode = ForwardMode.DECODE

    normal = PrefillAdder(
        allocator,
        None,
        running,
        max_prefill_tokens=8,
        chunked_prefill_size=None,
        new_token_ratio=0.5,
    )
    assert normal.budget.decode_reserved_tokens == 2

    active.retracted_stain = True
    conservative = PrefillAdder(
        allocator,
        None,
        running,
        max_prefill_tokens=8,
        chunked_prefill_size=None,
        new_token_ratio=0.5,
    )
    assert conservative.budget.decode_reserved_tokens == 3


def test_zero_extend_cache_hit_still_counts_as_an_accepted_request():
    allocator = TokenToKVPoolAllocator(2)
    cached_slots = allocator.alloc(2)
    assert cached_slots is not None
    cache = MiniRadixCache()
    cache.insert([1, 2], cached_slots)
    adder = PrefillAdder(
        allocator,
        cache,
        empty_batch(allocator, cache),
        max_prefill_tokens=2,
        chunked_prefill_size=None,
        new_token_ratio=0,
    )

    decisions = adder.add_requests(
        [req("hit", [1, 2], 1), req("next", [3], 1)], None, 2
    )

    assert [decision.result for decision in decisions] == [
        AddReqResult.ADMIT,
        AddReqResult.DEFER,
