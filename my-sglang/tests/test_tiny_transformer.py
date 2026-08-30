from __future__ import annotations

import numpy as np
import pytest

from my_sglang.models import (
    ForwardBatch,
    ForwardMode,
    Req,
    RequestStatus,
    SamplingParams,
)
from my_sglang.pools import ReqToTokenPool
from my_sglang.scheduler import MiniScheduler
from my_sglang.tiny_transformer import (
    TinyTransformerConfig,
    TinyTransformerRunner,
    apply_rope,
    rms_norm,
    stable_softmax,
)


def make_req(rid: str, ids: list[int], max_new_tokens: int = 2) -> Req:
    return Req(
        rid=rid,
        origin_input_ids=ids,
        sampling_params=SamplingParams(max_new_tokens=max_new_tokens),
    )


def attach(pool: ReqToTokenPool, req: Req) -> int:
    row = pool.alloc_one(req)
    req.req_pool_idx = row
    return row


def test_math_primitives_match_direct_definitions():
    values = np.asarray([3.0, 4.0], dtype=np.float32)
    weight = np.asarray([2.0, 0.5], dtype=np.float32)
    eps = 1e-5
    expected = values / np.sqrt(np.mean(values * values) + eps) * weight
    np.testing.assert_allclose(rms_norm(values, weight, eps), expected, rtol=1e-6)

    matrix = np.asarray([[1000.0, 1001.0, 999.0]], dtype=np.float32)
    shifted = matrix - np.max(matrix, axis=-1, keepdims=True)
    expected_softmax = np.exp(shifted) / np.sum(
        np.exp(shifted), axis=-1, keepdims=True
    )
    np.testing.assert_allclose(stable_softmax(matrix), expected_softmax, rtol=1e-6)


def test_rope_position_zero_is_identity_and_preserves_pair_norms():
    values = np.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    np.testing.assert_allclose(apply_rope(values, 0), values)

    rotated = apply_rope(values, 7)
    original_norms = values[:, 0::2] ** 2 + values[:, 1::2] ** 2
    rotated_norms = rotated[:, 0::2] ** 2 + rotated[:, 1::2] ** 2
    np.testing.assert_allclose(rotated_norms, original_norms, rtol=1e-6)


def test_single_token_forward_matches_independent_block_calculation():
    config = TinyTransformerConfig(
        vocab_size=16, hidden_size=8, num_heads=2, intermediate_size=12, seed=4
    )
    runner = TinyTransformerRunner(config)
    pool = ReqToTokenPool(1, 4)
    req = make_req("one", [3], max_new_tokens=1)
    row = attach(pool, req)
    pool.write(row, 0, [1])
    forward = ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        reqs=(req,),
        input_ids=(3,),
        req_pool_indices=(row,),
        out_cache_loc=(1,),
        seq_lens=(1,),
        extend_seq_lens=(1,),
        extend_range_starts=(0,),
    )

    actual = runner.run_batch(forward, pool)
    model = runner.model
    hidden = model.embedding[3].copy()
    mean_square = np.mean(hidden * hidden)
    normed = hidden / np.sqrt(mean_square + config.rms_norm_eps)
    # 一个 token 只能关注自己，所以每个 head 的 Softmax 权重必为 1。
    value = (normed @ model.wv).reshape(config.num_heads, config.head_dim)
    hidden = hidden + value.reshape(config.hidden_size) @ model.wo
    ffn_mean_square = np.mean(hidden * hidden)
    ffn_input = hidden / np.sqrt(ffn_mean_square + config.rms_norm_eps)
    gate = ffn_input @ model.w_gate
    gate = gate / (1.0 + np.exp(-gate))
    hidden = hidden + (gate * (ffn_input @ model.w_up)) @ model.w_down
    final = hidden / np.sqrt(np.mean(hidden * hidden) + config.rms_norm_eps)
    expected = int(np.argmax(final @ model.embedding.T))

    assert actual == [expected]
    assert model.kv_valid[1]
    assert model.hidden_valid[1]


def test_forward_batch_flattens_multiple_requests_and_writes_given_slots():
    runner = TinyTransformerRunner(
        TinyTransformerConfig(vocab_size=32, hidden_size=8, num_heads=2)
    )
    pool = ReqToTokenPool(2, 4)
    first = make_req("a", [1, 2])
    second = make_req("b", [3])
    row_a = attach(pool, first)
    row_b = attach(pool, second)
    pool.write(row_a, 0, [2, 5])
    pool.write(row_b, 0, [7])
    forward = ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        reqs=(first, second),
        input_ids=(1, 2, 3),
        req_pool_indices=(row_a, row_b),
        out_cache_loc=(2, 5, 7),
        seq_lens=(2, 1),
        extend_seq_lens=(2, 1),
        extend_range_starts=(0, 0),
    )

    sampled = runner.run_batch(forward, pool)

    assert len(sampled) == 2
    assert runner.model.kv_valid[[2, 5, 7]].all()
    assert "model:kv_read:pos=1:slots=2,5" in runner.trace
    assert "model:kv_read:pos=0:slots=7" in runner.trace


def test_decode_reads_history_and_overwrites_reused_slot():
    runner = TinyTransformerRunner(
        TinyTransformerConfig(vocab_size=32, hidden_size=8, num_heads=2)
    )
    pool = ReqToTokenPool(1, 4)
    req = make_req("decode", [1])
    row = attach(pool, req)
    pool.write(row, 0, [1])
    prefill = ForwardBatch(
        ForwardMode.EXTEND,
        (req,),
        (1,),
        (row,),
        (1,),
        (1,),
        extend_seq_lens=(1,),
        extend_range_starts=(0,),
    )
    first_token = runner.run_batch(prefill, pool)[0]

    pool.write(row, 1, [3])
    decode = ForwardBatch(
        ForwardMode.DECODE,
        (req,),
        (first_token,),
        (row,),
        (3,),
        (2,),
        extend_seq_lens=(1,),
        extend_range_starts=(1,),
    )
    runner.run_batch(decode, pool)
    old_key = runner.model.key_cache[3].copy()

    pool.write(row, 1, [3])
    replacement = ForwardBatch(
        ForwardMode.DECODE,
        (req,),
        ((first_token + 1) % 32,),
        (row,),
        (3,),
        (2,),
        extend_seq_lens=(1,),
        extend_range_starts=(1,),
    )
    runner.run_batch(replacement, pool)

    assert "model:kv_read:pos=1:slots=1,3" in runner.trace
    assert not np.allclose(runner.model.key_cache[3], old_key)


def test_full_prefix_hit_samples_from_cached_last_hidden():
    runner = TinyTransformerRunner(
        TinyTransformerConfig(vocab_size=32, hidden_size=8, num_heads=2)
    )
    pool = ReqToTokenPool(2, 4)
    original = make_req("original", [1, 2])
    original_row = attach(pool, original)
    pool.write(original_row, 0, [1, 2])
    prefill = ForwardBatch(
        ForwardMode.EXTEND,
        (original,),
        (1, 2),
        (original_row,),
        (1, 2),
        (2,),
        extend_seq_lens=(2,),
        extend_range_starts=(0,),
    )
    expected = runner.run_batch(prefill, pool)

    hit = make_req("hit", [1, 2])
    hit_row = attach(pool, hit)
    pool.write(hit_row, 0, [1, 2])
    full_hit = ForwardBatch(
        ForwardMode.EXTEND,
        (hit,),
        (),
        (hit_row,),
        (),
        (2,),
        extend_seq_lens=(0,),
        extend_range_starts=(2,),
    )

    assert runner.run_batch(full_hit, pool) == expected
    assert "model:full_prefix_hit:hit:slot=2" in runner.trace


def test_scheduler_tiny_model_closes_prefill_decode_lifecycle():
    runner = TinyTransformerRunner()
    scheduler = MiniScheduler(
        runner, max_running_reqs=2, max_total_tokens=8, trace=False
    )
    req = make_req("e2e", [1, 2], max_new_tokens=3)
    scheduler.add_request(req)

    scheduler.run_until_complete()

    assert req.output_ids == [94, 94, 94]
    assert req.status is RequestStatus.FINISHED
    assert scheduler.req_to_token_pool.active_count == 0
    assert scheduler.token_to_kv_pool_allocator.allocated_size == 0
    assert any(item.startswith("model:sample:extend:e2e") for item in runner.trace)
    assert sum(item.startswith("model:sample:decode:e2e") for item in runner.trace) == 2


def test_scheduler_tiny_model_supports_chunk_and_full_radix_hit():
    runner = TinyTransformerRunner()
    scheduler = MiniScheduler(
        runner,
        max_running_reqs=2,
        max_total_tokens=4,
        chunked_prefill_size=1,
        enable_radix_cache=True,
        new_token_ratio=0,
        trace=False,
    )
    first = make_req("first", [1, 2], max_new_tokens=1)
    scheduler.add_request(first)

    first_chunk = scheduler.step()
    assert first_chunk.batch is not None
    assert first.status is RequestStatus.PREFILLING
    assert first.output_ids == []
    scheduler.run_until_complete()
    expected = list(first.output_ids)

    hit = make_req("hit", [1, 2], max_new_tokens=1)
    scheduler.add_request(hit)
    result = scheduler.step()

    assert result.batch is not None
    assert result.batch.input_ids == ()
    assert hit.output_ids == expected
    assert any(item.startswith("model:full_prefix_hit:hit") for item in runner.trace)


def test_scheduler_tiny_model_reuses_partial_radix_kv_without_rewriting_prefix():
    runner = TinyTransformerRunner()
    scheduler = MiniScheduler(
        runner,
        max_running_reqs=2,
        max_total_tokens=4,
        enable_radix_cache=True,
        new_token_ratio=0,
        trace=False,
    )
    first = make_req("first", [1, 2], max_new_tokens=1)
    scheduler.add_request(first)
    first_result = scheduler.step()
    assert first_result.batch is not None
    cached_slots = first_result.batch.out_cache_loc
    cached_keys = runner.model.key_cache[list(cached_slots)].copy()

    reuse = make_req("reuse", [1, 2, 3], max_new_tokens=1)
    scheduler.add_request(reuse)
    reuse_result = scheduler.step()

    assert reuse_result.batch is not None
    assert reuse_result.batch.prefix_indices_by_req == (cached_slots,)
    assert reuse_result.batch.input_ids_by_req == ((3,),)
    np.testing.assert_array_equal(
        runner.model.key_cache[list(cached_slots)], cached_keys
    )
    new_slot = reuse_result.batch.out_cache_loc[0]
    history = ",".join(str(slot) for slot in (*cached_slots, new_slot))
    assert f"model:kv_read:pos=2:slots={history}" in runner.trace


def test_tiny_model_rejects_invalid_token_position_and_missing_cache():
    with pytest.raises(ValueError, match="even head dimension"):
        TinyTransformerConfig(hidden_size=6, num_heads=2)

    runner = TinyTransformerRunner(
        TinyTransformerConfig(
            vocab_size=8,
            hidden_size=8,
            num_heads=2,
            max_position_embeddings=1,
        )
    )
    pool = ReqToTokenPool(1, 3)
    req = make_req("bad", [8])
    row = attach(pool, req)
    pool.write(row, 0, [1])
    invalid_token = ForwardBatch(
        ForwardMode.EXTEND,
        (req,),
        (8,),
        (row,),
        (1,),
        (1,),
        extend_seq_lens=(1,),
        extend_range_starts=(0,),
    )
    with pytest.raises(ValueError, match="outside vocab_size"):
        runner.run_batch(invalid_token, pool)

    pool.write(row, 1, [2])
    invalid_position = ForwardBatch(
        ForwardMode.DECODE,
        (req,),
        (1,),
        (row,),
        (2,),
        (2,),
        extend_seq_lens=(1,),
        extend_range_starts=(1,),
    )
    with pytest.raises(ValueError, match="exceeds max_position_embeddings"):
        runner.run_batch(invalid_position, pool)

    full_hit = ForwardBatch(
        ForwardMode.EXTEND,
        (req,),
        (),
        (row,),
        (),
        (1,),
        extend_seq_lens=(0,),
        extend_range_starts=(1,),
    )
    with pytest.raises(RuntimeError, match="cached final hidden state"):
        runner.run_batch(full_hit, pool)
