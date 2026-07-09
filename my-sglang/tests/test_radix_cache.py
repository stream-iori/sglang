from __future__ import annotations

import pytest

from my_sglang.radix_cache import MiniRadixCache


def test_radix_cache_matches_and_splits_shared_prefix():
    cache = MiniRadixCache()

    first = cache.insert([1, 2, 3], [10, 11, 12])
    assert first.prefix_len == 0
    assert first.inserted_slots == (10, 11, 12)

    match = cache.match_prefix([1, 2, 9])
    assert match.token_count == 2
    assert match.slot_ids == (10, 11)

    second = cache.insert([1, 2, 4], [20, 21, 22])
    assert second.prefix_len == 2
    assert second.inserted_slots == (22,)
    assert cache.total_size() == 4

    assert cache.match_prefix([1, 2, 3, 5]).slot_ids == (10, 11, 12)
    assert cache.match_prefix([1, 2, 4, 5]).slot_ids == (10, 11, 22)


def test_radix_cache_rejects_mismatched_token_and_slot_lengths():
    cache = MiniRadixCache()

    with pytest.raises(ValueError, match="same length"):
        cache.insert([1, 2], [10])


def test_radix_cache_evicts_lru_leaf_when_capacity_is_exceeded():
    cache = MiniRadixCache(max_slots=2)

    first = cache.insert([1, 2], [10, 11])
    assert first.evicted_slots == ()
    assert cache.total_size() == 2

    second = cache.insert([3, 4], [30, 31])

    assert second.evicted_slots == (10, 11)
    assert cache.total_size() == 2
    assert cache.match_prefix([1, 2]).slot_ids == ()
    assert cache.match_prefix([3, 4]).slot_ids == (30, 31)


def test_radix_cache_does_not_evict_pinned_leaf():
    cache = MiniRadixCache(max_slots=2)
    cache.insert([1, 2], [10, 11])

    match = cache.match_prefix([1, 2], pin=True)
    result = cache.insert([3, 4], [30, 31])

    assert result.evicted_slots == (30, 31)
    assert cache.total_size() == 2
    assert cache.match_prefix([1, 2]).slot_ids == (10, 11)

    cache.release_nodes(match.matched_nodes)
