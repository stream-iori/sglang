from __future__ import annotations

from my_sglang.models import (
    FINISH_LENGTH,
    ForwardBatch,
    ForwardMode,
    Range,
    Req,
    SamplingParams,
)
from my_sglang.overlap_scheduler import FutureMap
from my_sglang.pools import ReqToTokenPool, TokenToKVPoolAllocator
from my_sglang.radix_cache import MiniRadixCache


def test_req_uses_srt_core_field_names_and_nesting():
    req = Req("r0", [1, 2], SamplingParams(max_new_tokens=1))
    req.extend_range = Range(0, 2)
    req.kv.kv_allocated_len = 2
    req.kv_committed_len = 2
    req.append_output(10)

    assert req.maybe_finish()
    assert req.extend_range.length == 2
    assert req.finished_reason == FINISH_LENGTH(1)
    assert req.full_untruncated_fill_ids == [1, 2, 10]
    assert req.get_fill_ids() == [1, 2, 10]
    assert not hasattr(req, "fill_len")
    assert not hasattr(req, "fill_ids")
    assert not hasattr(req, "extend_input_len")
    assert not hasattr(req, "kv_allocated_len")
    assert not hasattr(req, "finish_reason")


def test_forward_batch_uses_srt_core_field_names():
    req = Req("r0", [1], SamplingParams(max_new_tokens=1))
    batch = ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        reqs=(req,),
        input_ids=(1,),
        req_pool_indices=(1,),
        out_cache_loc=(1,),
        seq_lens=(1,),
        extend_seq_lens=(1,),
        extend_range_starts=(0,),
    )

    assert batch.forward_mode is ForwardMode.EXTEND
    assert batch.seq_lens_sum == 1
    assert not hasattr(batch, "mode")
    assert not hasattr(batch, "out_cache_locs")


def test_pool_cache_and_overlap_use_srt_core_field_names():
    req_pool = ReqToTokenPool(size=2, max_context_len=4)
    allocator = TokenToKVPoolAllocator(size=4)
    cache = MiniRadixCache()
    future_map = FutureMap(req_pool.req_to_token.shape[0])

    assert req_pool.size == 2
    assert len(req_pool.free_slots) == 2
    assert allocator.size == 4
    assert len(allocator.free_pages) == 4
    assert cache.root_node.key == ()
    assert cache.root_node.value == ()
    assert cache.root_node.lock_ref == 0
    assert future_map.output_tokens_buf.shape == (2,)
    assert not hasattr(req_pool, "capacity")
    assert not hasattr(cache, "root")
