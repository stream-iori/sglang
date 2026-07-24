from __future__ import annotations

import pytest

from my_sglang.models import ForwardMode, Req, RequestStatus, SamplingParams
from my_sglang.scheduler import MiniScheduler


class FakeRunner:
    """Deterministic runner: scheduler tests never need model weights."""

    def __init__(self, prefill_tokens=None, decode_tokens=None, extend_tokens=None):
        self.prefill_tokens = list(prefill_tokens or [100, 110, 120, 130])
        self.decode_tokens = list(decode_tokens or [101, 102, 103, 104])
        self.extend_tokens = list(extend_tokens or [201, 202, 203, 204])
        self.prefill_calls: list[dict] = []
        self.extend_calls: list[dict] = []
        self.decode_calls: list[list[str]] = []
        self.removed: list[str] = []

    def prefill(
        self,
        req_id,
        new_token_ids,
        full_token_ids,
        prefix_slot_ids,
        new_slot_ids,
        req_pool_idx,
    ):
        self.prefill_calls.append(
            {
                "req_id": req_id,
                "new_token_ids": list(new_token_ids),
                "full_token_ids": list(full_token_ids),
                "prefix_slot_ids": list(prefix_slot_ids),
                "new_slot_ids": list(new_slot_ids),
                "req_pool_idx": req_pool_idx,
            }
        )
        return self.prefill_tokens.pop(0)

    def extend(self, req_id, new_token_ids, new_slot_ids):
        self.extend_calls.append(
            {
                "req_id": req_id,
                "new_token_ids": list(new_token_ids),
                "new_slot_ids": list(new_slot_ids),
            }
        )
        return self.extend_tokens.pop(0)

    def decode_batch(self, req_ids):
        self.decode_calls.append(list(req_ids))
        return [self.decode_tokens.pop(0) for _ in req_ids]

    def remove_request(self, req_id):
        self.removed.append(req_id)


class FailingRunner(FakeRunner):
    def prefill(self, *args, **kwargs):
        raise RuntimeError("model failed")


def make_req(rid="r0", ids=None, max_new_tokens=2, eos=None):
    return Req(
        rid=rid,
        origin_input_ids=[1, 2] if ids is None else list(ids),
        sampling_params=SamplingParams(
            max_new_tokens=max_new_tokens,
            eos_token_ids=frozenset(eos or []),
        ),
    )


def test_single_request_extend_then_decode_lifecycle():
    runner = FakeRunner(prefill_tokens=[10], decode_tokens=[11])
    scheduler = MiniScheduler(
        runner, max_running_reqs=2, max_total_tokens=8
    )
    req = make_req(max_new_tokens=2)

    scheduler.add_request(req)
    extend = scheduler.step()

    assert extend.batch is not None and extend.batch.mode is ForwardMode.EXTEND
    assert extend.batch.input_ids_by_req == ((1, 2),)
    assert extend.batch.out_cache_locs == ((1, 2),)  # slot/page 0 is padding
    assert req.output_ids == [10]
    assert req.status is RequestStatus.RUNNING
    assert req.kv_allocated_len == req.kv_committed_len == 2

    decode = scheduler.step()

    assert decode.batch is not None and decode.batch.mode is ForwardMode.DECODE
    assert decode.finished_rids == ("r0",)
    assert req.output_ids == [10, 11]
    assert req.finish_reason == "length"
    assert scheduler.req_to_token_pool.active_count == 0
    assert scheduler.token_to_kv_pool_allocator.allocated_size == 0
    assert runner.removed == ["r0"]


def test_prefill_priority_means_one_forward_batch_per_step():
    runner = FakeRunner(prefill_tokens=[10, 20], decode_tokens=[11, 21])
    scheduler = MiniScheduler(runner, max_running_reqs=4, max_total_tokens=16)
    old = make_req("old", [1], max_new_tokens=2)
    new = make_req("new", [2], max_new_tokens=2)

    scheduler.add_request(old)
    scheduler.step()
    scheduler.add_request(new)
    second = scheduler.step()

    assert second.batch is not None and second.batch.mode is ForwardMode.EXTEND
    assert [req.rid for req in second.batch.reqs] == ["new"]
    assert runner.decode_calls == []

    third = scheduler.step()
    assert third.batch is not None and third.batch.mode is ForwardMode.DECODE
    assert [req.rid for req in third.batch.reqs] == ["old", "new"]
    assert third.finished_rids == ("old", "new")


def test_req_to_token_matrix_records_extend_and_decode_positions():
    scheduler = MiniScheduler(
        FakeRunner(prefill_tokens=[10], decode_tokens=[11]),
        max_running_reqs=2,
        max_total_tokens=8,
    )
    req = make_req("r0", [7, 8], max_new_tokens=3)

    scheduler.add_request(req)
    first = scheduler.step()
    assert first.batch is not None
    row = req.req_pool_idx
    assert row is not None
    assert tuple(scheduler.req_to_token_pool.row(row, 2)) == first.batch.out_cache_locs[0]

    second = scheduler.step()
    assert second.batch is not None
    assert scheduler.req_to_token_pool.get(row, 2) == second.batch.out_cache_locs[0][0]
    assert req.kv_allocated_len == req.kv_committed_len == 3


def test_eos_finishes_after_extend_and_releases_resources():
    scheduler = MiniScheduler(
        FakeRunner(prefill_tokens=[42]), max_running_reqs=2, max_total_tokens=8
    )
    req = make_req("r0", [1], max_new_tokens=8, eos={42})
    scheduler.add_request(req)

    result = scheduler.step()

    assert result.finished_rids == ("r0",)
    assert req.finish_reason == "eos"
    assert scheduler.memory_snapshot().allocated_tokens == 0


def test_duplicate_rid_and_empty_prompt_are_rejected():
    scheduler = MiniScheduler(FakeRunner())
    scheduler.add_request(make_req("same", [1]))
    with pytest.raises(ValueError, match="duplicate rid"):
        scheduler.add_request(make_req("same", [2]))
    with pytest.raises(ValueError, match="origin_input_ids"):
        make_req("empty", [])


def test_impossible_request_is_aborted_by_admission_not_half_allocated():
    scheduler = MiniScheduler(FakeRunner(), max_running_reqs=2, max_total_tokens=1)
    req = make_req("too-long", [1, 2])
    scheduler.add_request(req)

    result = scheduler.step()

    assert result.aborted_rids == ("too-long",)
    assert req.finish_reason == "abort"
    assert scheduler.req_to_token_pool.active_count == 0
    assert scheduler.token_to_kv_pool_allocator.allocated_size == 0


def test_runner_failure_rolls_back_allocated_but_uncommitted_kv():
    scheduler = MiniScheduler(FailingRunner(), max_running_reqs=2, max_total_tokens=4)
    req = make_req("r0", [1, 2])
    scheduler.add_request(req)

    with pytest.raises(RuntimeError, match="model failed"):
        scheduler.step()

    assert scheduler.waiting_queue == [req]
    assert req.req_pool_idx is None
    assert req.kv_allocated_len == req.kv_committed_len == 0
    assert scheduler.memory_snapshot().allocated_tokens == 0


def test_chunked_prefill_has_one_unfinished_request_and_no_early_output():
    runner = FakeRunner(
        prefill_tokens=[90], extend_tokens=[91, 10], decode_tokens=[11]
    )
    scheduler = MiniScheduler(
        runner,
        max_running_reqs=2,
        max_total_tokens=8,
        chunked_prefill_size=2,
    )
    req = make_req("r0", [1, 2, 3, 4, 5], max_new_tokens=2)
    scheduler.add_request(req)

    first = scheduler.step()
    assert first.batch is not None
    assert first.batch.input_ids_by_req == ((1, 2),)
    assert first.batch.is_last_prefill_chunk_by_req == (False,)
    assert req.status is RequestStatus.PREFILLING
    assert req.fill_len == req.kv_committed_len == 2
    assert req.output_ids == []

    second = scheduler.step()
    assert second.batch is not None
    assert second.batch.input_ids_by_req == ((3, 4),)
    assert req.fill_len == req.kv_committed_len == 4
    assert req.output_ids == []

    third = scheduler.step()
    assert third.batch is not None
    assert third.batch.input_ids_by_req == ((5,),)
    assert third.batch.is_last_prefill_chunk_by_req == (True,)
    assert req.output_ids == [10]
    assert req.status is RequestStatus.RUNNING

    fourth = scheduler.step()
    assert fourth.finished_rids == ("r0",)
    assert req.output_ids == [10, 11]


def test_unfinished_chunk_is_cached_only_at_complete_page_boundaries():
    scheduler = MiniScheduler(
        FakeRunner(prefill_tokens=[90], extend_tokens=[91, 10]),
        max_running_reqs=2,
        max_total_tokens=8,
        page_size=2,
        enable_radix_cache=True,
        chunked_prefill_size=2,
    )
    assert scheduler.tree_cache is not None
    req = make_req("r0", [1, 2, 3, 4, 5], max_new_tokens=1)
    scheduler.add_request(req)

    scheduler.step()
    assert req.cache_protected_len == 2
    assert scheduler.tree_cache.protected_size() == 2
    scheduler.step()
    assert req.cache_protected_len == 4
    assert scheduler.tree_cache.protected_size() == 4
    scheduler.step()

    assert req.status is RequestStatus.FINISHED
    assert scheduler.tree_cache.total_size() == 4
    assert scheduler.tree_cache.evictable_size() == 4
    assert scheduler.token_to_kv_pool_allocator.allocated_size == 4


def test_radix_cache_reuses_prefix_and_evicts_lru_for_new_admission():
    runner = FakeRunner(prefill_tokens=[10, 20, 30])
    scheduler = MiniScheduler(
        runner,
        max_running_reqs=2,
        max_total_tokens=4,
        enable_radix_cache=True,
        new_token_ratio=0,
    )
    first = make_req("first", [1, 2], max_new_tokens=1)
    scheduler.add_request(first)
    first_result = scheduler.step()
    assert first_result.batch is not None
    cached_slots = first_result.batch.out_cache_locs[0]

    reuse = make_req("reuse", [1, 2, 3], max_new_tokens=1)
    scheduler.add_request(reuse)
    reuse_result = scheduler.step()
    assert reuse_result.batch is not None
    assert reuse_result.batch.prefix_slot_ids_by_req == (cached_slots,)
    assert reuse_result.batch.input_ids_by_req == ((3,),)

    replacement = make_req("replacement", [7, 8, 9, 10], max_new_tokens=1)
    scheduler.add_request(replacement)
    scheduler.step()
    assert scheduler.tree_cache is not None
    assert scheduler.tree_cache.match_prefix([1, 2], pin=False).token_count == 0
    assert scheduler.tree_cache.match_prefix([7, 8, 9, 10], pin=False).token_count == 4


def test_radix_full_prompt_hit_allocates_no_extend_slots():
    runner = FakeRunner(prefill_tokens=[10, 20])
    scheduler = MiniScheduler(
        runner,
        max_running_reqs=2,
        max_total_tokens=2,
        enable_radix_cache=True,
        new_token_ratio=0,
    )
    assert scheduler.tree_cache is not None
    scheduler.add_request(make_req("first", [1, 2], max_new_tokens=1))
    first = scheduler.step()
    assert first.batch is not None
    cached_slots = first.batch.out_cache_locs[0]

    scheduler.add_request(make_req("hit", [1, 2], max_new_tokens=1))
    hit = scheduler.step()
    assert hit.batch is not None

    assert hit.batch.prefix_slot_ids_by_req == (cached_slots,)
    assert hit.batch.input_ids_by_req == ((),)
    assert hit.batch.out_cache_locs == ((),)
    assert runner.prefill_calls[1]["new_slot_ids"] == []
    assert scheduler.token_to_kv_pool_allocator.allocated_size == 2


def test_decode_pressure_retracts_one_request_then_readmits_it():
    runner = FakeRunner(
        prefill_tokens=[10, 20, 30], decode_tokens=[11, 21, 22]
    )
    scheduler = MiniScheduler(
        runner,
        max_running_reqs=2,
        max_total_tokens=4,
        new_token_ratio=0,
    )
    a = make_req("a", [1], max_new_tokens=3)
    b = make_req("b", [2], max_new_tokens=3)
    scheduler.add_request(a)
    scheduler.add_request(b)
    scheduler.step()  # two prompt slots
    scheduler.step()  # two decode slots; pool is now full

    pressure = scheduler.step()
    assert pressure.retracted_rids == ("a",)
    assert pressure.finished_rids == ("b",)
    assert a.status is RequestStatus.WAITING
    assert a.retracted_stain is True
    assert a.output_ids == [10, 11]

    readmit = scheduler.step()
    assert readmit.batch is not None
    assert readmit.batch.mode is ForwardMode.EXTEND
    assert readmit.finished_rids == ("a",)
    assert a.output_ids == [10, 11, 30]


def test_last_decode_request_is_aborted_when_no_page_can_be_reclaimed():
    scheduler = MiniScheduler(
        FakeRunner(prefill_tokens=[10]),
        max_running_reqs=1,
        max_total_tokens=1,
    )
    req = make_req("r0", [1], max_new_tokens=2)
    scheduler.add_request(req)
    scheduler.step()

    result = scheduler.step()

    assert result.aborted_rids == ("r0",)
    assert req.finish_reason == "abort"
    assert scheduler.memory_snapshot().allocated_tokens == 0


def test_max_prefill_tokens_admits_fcfs_subset():
    scheduler = MiniScheduler(
        FakeRunner(prefill_tokens=[10]),
        max_running_reqs=3,
        max_total_tokens=12,
        max_prefill_tokens=2,
        new_token_ratio=0,
    )
    first = make_req("first", [1, 2], max_new_tokens=2)
    second = make_req("second", [3, 4], max_new_tokens=2)
    scheduler.add_request(first)
    scheduler.add_request(second)

    result = scheduler.step()

    assert result.batch is not None
    assert result.memory is not None
    assert [req.rid for req in result.batch.reqs] == ["first"]
    assert scheduler.waiting_queue == [second]
    assert result.memory.decode_reserved_tokens == 0
