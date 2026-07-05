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
