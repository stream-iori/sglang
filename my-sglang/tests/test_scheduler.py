from __future__ import annotations

import pytest

from my_sglang.models import Req, RequestStatus, SamplingParams
from my_sglang.radix_cache import MiniRadixCache
from my_sglang.scheduler import MiniScheduler


class FakeRunner:
    # FakeRunner 不加载真实模型，只按预设 token 返回结果。
    # 这样单元测试可以专注验证 scheduler 状态机，不受 MLX 和模型权重影响。
    def __init__(self, prefill_tokens=None, decode_tokens=None, extend_tokens=None):
        self.prefill_tokens = list(prefill_tokens or [100])
        self.decode_tokens = list(decode_tokens or [101, 102, 103])
        self.extend_tokens = list(extend_tokens or [201, 202, 203])
        self.prefill_calls = []
        self.extend_calls = []
        self.decode_calls = []
        self.removed = []
        self.live = set()

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
        self.live.add(req_id)
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
        self.live.discard(req_id)


def make_req(rid="r0", ids=None, max_new_tokens=2, eos=None):
    # 测试辅助函数：用最少字段构造一个 Req。
    origin_ids = [1, 2] if ids is None else list(ids)
    return Req(
        rid=rid,
        origin_input_ids=origin_ids,
        sampling_params=SamplingParams(
            max_new_tokens=max_new_tokens,
            eos_token_ids=frozenset(eos or []),
        ),
    )


def test_single_request_prefill_then_decode_to_length_finish():
    # 单请求完整生命周期：prefill 产生第 1 个 token，下一轮 decode 产生第 2 个 token 后结束。
    runner = FakeRunner(prefill_tokens=[10], decode_tokens=[11])
    scheduler = MiniScheduler(runner, max_running_reqs=2, max_total_tokens=8)
    req = make_req(max_new_tokens=2)

    scheduler.add_request(req)
    first = scheduler.step()

    assert first.prefill_batch is not None
    assert first.prefill_batch.mode == "prefill"
    assert first.decode_batch is None
    assert req.output_ids == [10]
    assert req.status is RequestStatus.RUNNING
    assert scheduler.req_pool.active_count == 1
    assert scheduler.kv_pool.active_count == 2
    assert runner.decode_calls == []

    second = scheduler.step()

    assert second.decode_batch is not None
    assert second.finished_rids == ("r0",)
    assert req.output_ids == [10, 11]
    assert req.status is RequestStatus.FINISHED
    assert req.finish_reason == "length"
    assert scheduler.req_pool.active_count == 0
    assert scheduler.kv_pool.active_count == 0
    assert scheduler.req_to_token.size == 0
    assert runner.removed == ["r0"]


def test_prefill_batches_multiple_waiting_requests():
    # 多个 waiting 请求应该在同一轮被拼成一个 prefill batch。
    runner = FakeRunner(prefill_tokens=[10, 20], decode_tokens=[11, 21])
    scheduler = MiniScheduler(runner, max_running_reqs=4, max_total_tokens=16)
    r0 = make_req("r0", [1, 2], max_new_tokens=2)
    r1 = make_req("r1", [3], max_new_tokens=2)

    scheduler.add_request(r0)
    scheduler.add_request(r1)
    result = scheduler.step()

    assert result.prefill_batch is not None
    assert [req.rid for req in result.prefill_batch.reqs] == ["r0", "r1"]
    assert result.prefill_batch.input_ids_by_req == ((1, 2), (3,))
    assert result.prefill_batch.seq_lens == (2, 1)
    assert result.decode_batch is None
    assert [req.rid for req in scheduler.running_reqs] == ["r0", "r1"]


def test_new_prefill_is_not_decoded_in_same_step_as_old_running_request():
    # 本轮刚 prefill 的请求不会马上 decode；只有本轮开始前已 running 的请求会 decode。
    runner = FakeRunner(prefill_tokens=[10, 20], decode_tokens=[11, 12, 21])
    scheduler = MiniScheduler(runner, max_running_reqs=4, max_total_tokens=16)
    r0 = make_req("r0", [1], max_new_tokens=3)
    r1 = make_req("r1", [2], max_new_tokens=2)

    scheduler.add_request(r0)
    scheduler.step()
    scheduler.add_request(r1)
    result = scheduler.step()

    assert result.prefill_batch is not None
    assert [req.rid for req in result.prefill_batch.reqs] == ["r1"]
    assert result.decode_batch is not None
    assert [req.rid for req in result.decode_batch.reqs] == ["r0"]
    assert runner.decode_calls == [["r0"]]
    assert r1.output_ids == [20]


def test_req_to_token_records_prompt_and_decode_input_slots():
    # 验证 req_to_token 映射：序列位置应该能找到对应的 KV slot。
    runner = FakeRunner(prefill_tokens=[10], decode_tokens=[11])
    scheduler = MiniScheduler(runner, max_running_reqs=2, max_total_tokens=8)
    req = make_req("r0", [7, 8], max_new_tokens=3)

    scheduler.add_request(req)
    scheduler.step()
    req_idx = req.req_pool_idx

    assert req_idx is not None
    prompt_slots = list(req.kv_slots)
    assert scheduler.req_to_token.get(req_idx, 0) == prompt_slots[0]
    assert scheduler.req_to_token.get(req_idx, 1) == prompt_slots[1]

    scheduler.step()

    assert scheduler.req_to_token.get(req_idx, 2) == req.kv_slots[-1]


def test_eos_finishes_after_prefill_and_releases_resources():
    # prefill 直接生成 eos 时，请求应立即结束并释放资源，不进入 running 队列。
    runner = FakeRunner(prefill_tokens=[42])
    scheduler = MiniScheduler(runner, max_running_reqs=2, max_total_tokens=8)
    req = make_req("r0", [1], max_new_tokens=8, eos={42})

    scheduler.add_request(req)
    result = scheduler.step()

    assert result.finished_rids == ("r0",)
    assert req.status is RequestStatus.FINISHED
    assert req.finish_reason == "eos"
    assert scheduler.running_reqs == []
    assert scheduler.kv_pool.active_count == 0
    assert runner.removed == ["r0"]


def test_duplicate_rid_rejected():
    # 同一个 rid 不能重复进入调度器，否则资源归属会不清晰。
    scheduler = MiniScheduler(FakeRunner())
    scheduler.add_request(make_req("same", [1]))

    with pytest.raises(ValueError, match="duplicate rid"):
        scheduler.add_request(make_req("same", [2]))


def test_empty_prompt_rejected():
    # mini runtime 暂不支持空 prompt；真实系统一般会在更早的 tokenizer/输入校验层处理。
    with pytest.raises(ValueError, match="origin_input_ids"):
        make_req("r0", [])


def test_kv_exhaustion_rolls_back_waiting_state():
    # 资源不足时不能留下半分配状态：请求应回到 waiting，ReqPool/KVPool 都要回滚。
    scheduler = MiniScheduler(FakeRunner(), max_running_reqs=2, max_total_tokens=1)
    req = make_req("r0", [1, 2])
    scheduler.add_request(req)

    with pytest.raises(RuntimeError, match="KVPool exhausted"):
        scheduler.step()

    assert scheduler.waiting_queue == [req]
    assert scheduler.req_pool.active_count == 0
    assert scheduler.kv_pool.active_count == 0
    assert req.req_pool_idx is None
    assert req.kv_slots == []
    assert req.prefix_slot_ids == []
    assert req.owned_kv_slots == []


def test_chunked_prefill_processes_prompt_across_multiple_steps():
    runner = FakeRunner(
        prefill_tokens=[90],
        extend_tokens=[91, 10],
        decode_tokens=[11],
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

    assert first.prefill_batch is not None
    assert first.prefill_batch.input_ids_by_req == ((1, 2),)
    assert first.prefill_batch.chunk_starts_by_req == (0,)
    assert first.prefill_batch.is_last_prefill_chunk_by_req == (False,)
    assert req.status is RequestStatus.PREFILLING
    assert req.prefill_pos == 2
    assert req.output_ids == []
    assert scheduler.running_reqs == []
    assert runner.extend_calls == []

    second = scheduler.step()

    assert second.prefill_batch is not None
    assert second.prefill_batch.input_ids_by_req == ((3, 4),)
    assert second.prefill_batch.chunk_starts_by_req == (2,)
    assert second.prefill_batch.is_last_prefill_chunk_by_req == (False,)
    assert req.status is RequestStatus.PREFILLING
    assert req.prefill_pos == 4
    assert req.output_ids == []
    assert runner.extend_calls[0]["new_token_ids"] == [3, 4]

    third = scheduler.step()

    assert third.prefill_batch is not None
    assert third.prefill_batch.input_ids_by_req == ((5,),)
    assert third.prefill_batch.chunk_starts_by_req == (4,)
    assert third.prefill_batch.is_last_prefill_chunk_by_req == (True,)
    assert req.status is RequestStatus.RUNNING
    assert req.prefill_pos == 5
    assert req.output_ids == [10]
    assert [running.rid for running in scheduler.running_reqs] == ["r0"]

    fourth = scheduler.step()

    assert fourth.decode_batch is not None
    assert fourth.finished_rids == ("r0",)
    assert req.output_ids == [10, 11]
    assert req.status is RequestStatus.FINISHED
    assert scheduler.prefilling_reqs == []
    assert scheduler.req_pool.active_count == 0
    assert scheduler.kv_pool.active_count == 0


def test_chunked_prefill_can_run_in_same_step_as_decode():
    runner = FakeRunner(
        prefill_tokens=[10, 20],
        extend_tokens=[21],
        decode_tokens=[11, 12],
    )
    scheduler = MiniScheduler(
        runner,
        max_running_reqs=4,
        max_total_tokens=16,
        chunked_prefill_size=2,
    )
    old_req = make_req("old", [9], max_new_tokens=3)
    new_req = make_req("new", [1, 2, 3], max_new_tokens=2)

    scheduler.add_request(old_req)
    scheduler.step()
    scheduler.add_request(new_req)
    result = scheduler.step()

    assert result.prefill_batch is not None
    assert result.decode_batch is not None
    assert [req.rid for req in result.prefill_batch.reqs] == ["new"]
    assert result.prefill_batch.input_ids_by_req == ((1, 2),)
    assert result.prefill_batch.is_last_prefill_chunk_by_req == (False,)
    assert [req.rid for req in result.decode_batch.reqs] == ["old"]
    assert new_req.status is RequestStatus.PREFILLING
    assert new_req.output_ids == []
    assert old_req.output_ids == [10, 11]


def test_chunked_prefill_reuses_radix_prefix_before_chunking_suffix():
    runner = FakeRunner(
        prefill_tokens=[10, 20],
        extend_tokens=[11, 21],
    )
    scheduler = MiniScheduler(
        runner,
        max_running_reqs=2,
        max_total_tokens=4,
        enable_radix_cache=True,
        chunked_prefill_size=1,
    )
    first = make_req("first", [1, 2], max_new_tokens=1)
    second = make_req("second", [1, 2, 3, 4], max_new_tokens=1)

    scheduler.add_request(first)
    first_chunk = scheduler.step()
    first_last = scheduler.step()
    cached_prefix_slots = (
        first_chunk.prefill_batch.out_cache_locs[0]
        + first_last.prefill_batch.out_cache_locs[0]
    )
    assert first.status is RequestStatus.FINISHED

    scheduler.add_request(second)
    second_first = scheduler.step()

    assert second_first.prefill_batch is not None
    assert second_first.prefill_batch.prefix_slot_ids_by_req == (cached_prefix_slots,)
    assert second_first.prefill_batch.input_ids_by_req == ((3,),)
    assert second_first.prefill_batch.chunk_starts_by_req == (2,)
    assert second_first.prefill_batch.is_last_prefill_chunk_by_req == (False,)
    assert second.status is RequestStatus.PREFILLING
    assert second.prefill_pos == 3
    assert runner.prefill_calls[1]["prefix_slot_ids"] == list(cached_prefix_slots)
    assert runner.prefill_calls[1]["new_token_ids"] == [3]

    second_last = scheduler.step()

    assert second_last.prefill_batch is not None
    assert second_last.prefill_batch.input_ids_by_req == ((4,),)
    assert second_last.prefill_batch.chunk_starts_by_req == (3,)
    assert second_last.prefill_batch.is_last_prefill_chunk_by_req == (True,)
    assert runner.extend_calls[1]["new_token_ids"] == [4]
    assert second.status is RequestStatus.FINISHED
    assert scheduler.kv_pool.active_count == 4
    assert scheduler.radix_cache is not None
    assert scheduler.radix_cache.total_size() == 4


def test_chunked_prefill_allocation_failure_rolls_back_waiting_request():
    scheduler = MiniScheduler(
        FakeRunner(),
        max_running_reqs=2,
        max_total_tokens=1,
        chunked_prefill_size=2,
    )
    req = make_req("r0", [1, 2])
    scheduler.add_request(req)

    with pytest.raises(RuntimeError, match="KVPool exhausted"):
        scheduler.step()

    assert scheduler.waiting_queue == [req]
    assert scheduler.prefilling_reqs == []
    assert scheduler.req_pool.active_count == 0
    assert scheduler.kv_pool.active_count == 0
    assert req.status is RequestStatus.WAITING
    assert req.prefill_pos == 0
    assert req.req_pool_idx is None
    assert req.kv_slots == []
    assert req.prefix_slot_ids == []
    assert req.owned_kv_slots == []


def test_radix_cache_reuses_prompt_prefix_slots_for_prefill_suffix():
    runner = FakeRunner(prefill_tokens=[10, 20])
    scheduler = MiniScheduler(
        runner,
        max_running_reqs=2,
        max_total_tokens=4,
        enable_radix_cache=True,
    )
    first = make_req("first", [1, 2], max_new_tokens=1)
    second = make_req("second", [1, 2, 3], max_new_tokens=1)

    scheduler.add_request(first)
    first_result = scheduler.step()
    cached_prefix_slots = first_result.prefill_batch.out_cache_locs[0]

    assert first.status is RequestStatus.FINISHED
    assert scheduler.kv_pool.active_count == 2
    assert scheduler.radix_cache is not None
    assert scheduler.radix_cache.total_size() == 2

    scheduler.add_request(second)
    second_result = scheduler.step()

    assert second_result.prefill_batch is not None
    assert second_result.prefill_batch.prefix_slot_ids_by_req == (cached_prefix_slots,)
    assert second_result.prefill_batch.input_ids_by_req == ((3,),)
    assert len(second_result.prefill_batch.out_cache_locs[0]) == 1
    assert runner.prefill_calls[1]["prefix_slot_ids"] == list(cached_prefix_slots)
    assert runner.prefill_calls[1]["new_token_ids"] == [3]
    assert second.status is RequestStatus.FINISHED
    assert scheduler.kv_pool.active_count == 3
    assert scheduler.radix_cache.total_size() == 3
    assert scheduler.req_pool.active_count == 0
    assert scheduler.req_to_token.size == 0
    assert second.kv_slots == []
    assert second.owned_kv_slots == []


def test_radix_cache_full_prompt_hit_allocates_no_new_prefill_slots():
    runner = FakeRunner(prefill_tokens=[10, 20])
    scheduler = MiniScheduler(
        runner,
        max_running_reqs=2,
        max_total_tokens=2,
        enable_radix_cache=True,
    )

    scheduler.add_request(make_req("first", [1, 2], max_new_tokens=1))
    scheduler.step()
    assert scheduler.kv_pool.active_count == 2

    scheduler.add_request(make_req("second", [1, 2], max_new_tokens=1))
    result = scheduler.step()

    assert result.prefill_batch is not None
    assert result.prefill_batch.input_ids_by_req == ((),)
    assert result.prefill_batch.out_cache_locs == ((),)
    assert runner.prefill_calls[1]["prefix_slot_ids"] == [0, 1]
    assert runner.prefill_calls[1]["new_token_ids"] == []
    assert runner.prefill_calls[1]["new_slot_ids"] == []
    assert scheduler.kv_pool.active_count == 2
    assert scheduler.radix_cache is not None
    assert scheduler.radix_cache.total_size() == 2


def test_radix_cache_lru_eviction_releases_kv_pool_slots():
    scheduler = MiniScheduler(
        FakeRunner(prefill_tokens=[10, 20]),
        max_running_reqs=2,
        max_total_tokens=4,
        radix_cache=MiniRadixCache(max_slots=2),
    )

    scheduler.add_request(make_req("first", [1, 2], max_new_tokens=1))
    scheduler.step()
    assert scheduler.kv_pool.active_count == 2
    assert scheduler.radix_cache is not None
    assert scheduler.radix_cache.match_prefix([1, 2]).slot_ids == (0, 1)

    scheduler.add_request(make_req("second", [3, 4], max_new_tokens=1))
    scheduler.step()

    assert scheduler.kv_pool.active_count == 2
    assert scheduler.radix_cache.total_size() == 2
    assert scheduler.radix_cache.match_prefix([1, 2]).slot_ids == ()
    assert scheduler.radix_cache.match_prefix([3, 4]).slot_ids == (2, 3)


def test_radix_cache_suffix_allocation_failure_rolls_back_request_only_slots():
    scheduler = MiniScheduler(
        FakeRunner(),
        max_running_reqs=2,
        max_total_tokens=3,
        enable_radix_cache=True,
    )
    first = make_req("first", [1, 2], max_new_tokens=1)
    second = make_req("second", [1, 2, 3, 4], max_new_tokens=1)

    scheduler.add_request(first)
    scheduler.step()
    assert scheduler.kv_pool.active_count == 2

    scheduler.add_request(second)
    with pytest.raises(RuntimeError, match="KVPool exhausted"):
        scheduler.step()

    assert scheduler.waiting_queue == [second]
    assert scheduler.req_pool.active_count == 0
    assert scheduler.kv_pool.active_count == 2
    assert second.req_pool_idx is None
    assert second.kv_slots == []
    assert second.prefix_slot_ids == []
    assert second.owned_kv_slots == []
