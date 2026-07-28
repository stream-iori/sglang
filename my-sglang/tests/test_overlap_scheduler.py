"""Overlap scheduler 的可执行教程。

推荐不要一开始跑整个文件，而是按下面顺序单测并断点观察：

1. ``test_launch_allocates_and_finalize_commits_prefill_and_decode``
   先理解手动模式的 launch/finalize 事务边界。
2. ``test_pipeline_launches_chained_decode_before_processing_previous_result``
   再理解生产流水的核心时序：launch B1 发生在 finalize B0 之前。
3. ``test_pipeline_new_prefill_breaks_decode_chain_before_launching_more_decode``
   理解为什么不是每一轮 decode 都能继续 chain。
4. ``test_pipeline_drops_already_launched_token_and_defers_physical_release``
   理解逻辑完成与 KV/request row 物理释放为什么必须分离。
5. ``test_pipeline_finalize_failure_discards_dependencies_and_requeues_request``
   最后理解整条依赖链的失败恢复。

FakeLazyRunner 不执行模型，只记录 start/kick/finalize 的调用顺序并返回固定 token。
因此这些测试关注的全部是 scheduler 状态机，可以在 CPU 上快速、确定地运行。
"""

from __future__ import annotations

import pytest
from typing import Any

from my_sglang.models import ForwardMode, Req, RequestStatus, SamplingParams
from my_sglang.overlap_scheduler import MiniOverlapScheduler


class FakeLazyRunner:
    """记录 start -> kick -> finalize 边界的确定性教学 runner。"""

    def __init__(
        self,
        prefill_tokens=None,
        decode_tokens=None,
        extend_tokens=None,
        *,
        fail_decode_finalize_at=None,
    ):
        self.prefill_tokens = list(prefill_tokens or [100, 110])
        self.decode_tokens = list(decode_tokens or [101, 102, 103])
        self.extend_tokens = list(extend_tokens or [201, 202, 203])
        self.events: list[str] = []
        self.fail_decode_finalize_at = fail_decode_finalize_at
        self.decode_finalize_count = 0

    def prefill(self, *args, **kwargs):
        handle = self.prefill_start(*args, **kwargs)
        self.prefill_kick(handle)
        return self.prefill_finalize(handle)

    def extend(self, *args, **kwargs):
        handle = self.extend_start(*args, **kwargs)
        self.extend_kick(handle)
        return self.extend_finalize(handle)

    def decode_batch(self, req_ids):
        handle = self.decode_batch_start(req_ids)
        self.decode_batch_kick(handle)
        return self.decode_batch_finalize(handle)

    def prefill_start(
        self,
        req_id,
        new_token_ids,
        full_token_ids,
        prefix_slot_ids,
        new_slot_ids,
        req_pool_idx,
    ):
        self.events.append(f"prefill_start:{req_id}")
        return {"rid": req_id, "token": self.prefill_tokens.pop(0)}

    def prefill_kick(self, pending: Any) -> None:
        self.events.append(f"prefill_kick:{pending['rid']}")

    def prefill_finalize(self, pending: Any) -> int:
        self.events.append(f"prefill_finalize:{pending['rid']}")
        return pending["token"]

    def extend_start(self, req_id, new_token_ids, new_slot_ids):
        self.events.append(f"extend_start:{req_id}")
        return {"rid": req_id, "token": self.extend_tokens.pop(0)}

    def extend_kick(self, pending: Any) -> None:
        self.events.append(f"extend_kick:{pending['rid']}")

    def extend_finalize(self, pending: Any) -> int:
        self.events.append(f"extend_finalize:{pending['rid']}")
        return pending["token"]

    def decode_batch_start(self, req_ids):
        key = ",".join(req_ids)
        self.events.append(f"decode_start:{key}")
        return {
            "rids": list(req_ids),
            "tokens": [self.decode_tokens.pop(0) for _ in req_ids],
        }

    def decode_batch_start_chained(self, previous):
        key = ",".join(previous["rids"])
        self.events.append(f"decode_chain_start:{key}")
        return {
            "rids": list(previous["rids"]),
            "tokens": [self.decode_tokens.pop(0) for _ in previous["rids"]],
        }

    def decode_batch_kick(self, pending: Any) -> None:
        self.events.append(f"decode_kick:{','.join(pending['rids'])}")

    def decode_batch_finalize(self, pending: Any) -> list[int]:
        self.events.append(f"decode_finalize:{','.join(pending['rids'])}")
        self.decode_finalize_count += 1
        if self.decode_finalize_count == self.fail_decode_finalize_at:
            raise RuntimeError("injected decode finalize failure")
        return pending["tokens"]

    def discard_pending(self, pending: Any) -> None:
        key = pending.get("rid") or ",".join(pending.get("rids", []))
        self.events.append(f"discard:{key}")

    def remove_request(self, req_id):
        self.events.append(f"remove:{req_id}")


def make_req(rid="r0", ids=None, max_new_tokens=2):
    return Req(
        rid=rid,
        origin_input_ids=[1, 2] if ids is None else list(ids),
        sampling_params=SamplingParams(max_new_tokens=max_new_tokens),
    )


def test_launch_allocates_and_finalize_commits_prefill_and_decode():
    # Given：一个 prompt 长度为 2、最多生成 2 token 的请求。
    runner = FakeLazyRunner(prefill_tokens=[10], decode_tokens=[11])
    scheduler = MiniOverlapScheduler(runner, max_running_reqs=2, max_total_tokens=8)
    req = make_req()

    #最终只是进入队列
    scheduler.add_request(req)

    # When：launch 只启动 prefill。Then：KV 已分配，但 CPU 输出和 committed
    # 水位都没有推进。这是观察 manual transaction 最清楚的断点。
    launched = scheduler.launch_step()
    assert launched.batch is not None
    assert launched.batch.mode is ForwardMode.EXTEND
    assert req.kv_allocated_len == 2
    assert req.kv_committed_len == 0
    assert req.output_ids == []
    assert runner.events == ["prefill_start:r0", "prefill_kick:r0"]

    # finalize 才把 runner token=10 写入 req，并确认前面分配的 KV。
    first = scheduler.finalize_pending()
    assert first.batch is not None
    assert first.batch.mode is ForwardMode.EXTEND
    assert req.kv_committed_len == 2
    assert req.output_ids == [10]

    scheduler.launch_step()
    assert req.kv_allocated_len == 3
    assert req.kv_committed_len == 2
    second = scheduler.finalize_pending()
    assert second.finished_rids == ("r0",)
    assert req.status is RequestStatus.FINISHED
    assert runner.events[-4:] == [
        "decode_start:r0",
        "decode_kick:r0",
        "decode_finalize:r0",
        "remove:r0",
    ]


def test_overlap_api_enforces_one_pending_batch():
    scheduler = MiniOverlapScheduler(FakeLazyRunner(), max_total_tokens=8)
    with pytest.raises(RuntimeError, match="launch_step"):
        scheduler.finalize_pending()
    scheduler.add_request(make_req())
    scheduler.launch_step()
    with pytest.raises(RuntimeError, match="finalize_pending"):
        scheduler.launch_step()


def test_overlap_supports_chunked_radix_and_paged_allocator_together():
    runner = FakeLazyRunner(
        prefill_tokens=[90], extend_tokens=[91, 10], decode_tokens=[11]
    )
    scheduler = MiniOverlapScheduler(
        runner,
        max_running_reqs=2,
        max_total_tokens=8,
        page_size=2,
        enable_radix_cache=True,
        chunked_prefill_size=2,
    )
    req = make_req("combo", [1, 2, 3, 4, 5], max_new_tokens=2)
    scheduler.add_request(req)

    for committed in (2, 4, 5):
        scheduler.launch_step()
        assert req.kv_allocated_len == committed
        assert req.kv_committed_len < req.kv_allocated_len
        scheduler.finalize_pending()
        assert req.kv_committed_len == committed

    assert req.output_ids == [10]
    assert req.cache_protected_len == 4
    scheduler.launch_step()
    assert req.kv_allocated_len == 6  # decode reuses slot 7 in the prompt tail page
    scheduler.finalize_pending()

    assert req.status is RequestStatus.FINISHED
    assert req.output_ids == [10, 11]
    assert scheduler.tree_cache is not None
    assert scheduler.tree_cache.total_size() == 6
    assert "extend_start:combo" in runner.events


def test_pipeline_launches_chained_decode_before_processing_previous_result():
    """主教材：用事件顺序证明这里存在真正的 B0/B1 overlap。"""

    # 固定 token 让每一次状态变化都可预测，max_new_tokens=4 保证至少出现
    # 两个连续 decode batch，能够构造 B0 -> B1 的设备侧依赖。
    runner = FakeLazyRunner(
        prefill_tokens=[10], decode_tokens=[11, 12, 13, 14]
    )
    scheduler = MiniOverlapScheduler(runner, max_running_reqs=2, max_total_tokens=8)
    req = make_req(max_new_tokens=4)
    scheduler.add_request(req)

    # Turn 1（冷启动）：result_queue 为空，只发射 prefill B0，不读取结果。
    warmup = scheduler.pipeline_step()
    assert warmup.queue_depth == 1
    assert warmup.launched_batches[0].mode is ForwardMode.EXTEND
    assert req.kv_allocated_len == req.kv_committed_len == 2
    assert req.output_ids == []

    # Turn 2：prefill 不能作为 decode chain 的前驱，先 finalize prefill，
    # 然后从普通调度入口发射首个 decode B0。
    transition = scheduler.pipeline_step()
    assert transition.barrier_reason == "non_decode"
    assert req.output_ids == [10]
    assert transition.queue_depth == 1
    assert transition.queue[0].mode is ForwardMode.DECODE

    # Turn 3（稳定态）：先用 B0 的 future token 发射 B1，再 finalize B0。
    # 下面的顺序断言是整个 production-shaped overlap 最关键的测试证据。
    steady = scheduler.pipeline_step()
    chain_index = runner.events.index("decode_chain_start:r0")
    first_finalize_index = runner.events.index("decode_finalize:r0")
    assert chain_index < first_finalize_index
    assert steady.queue_depth == 1
    assert steady.queue[0].kind == "chained"
    # chained decode 没有伪造 CPU input id；它携带 FutureTokenRef，让 runner
    # 直接从前驱 lazy handle 的对应输出行取得输入。
    assert steady.launched_batches[0].input_ids_by_req == ((),)
    assert steady.launched_batches[0].input_future_refs_by_req[0] is not None
    assert req.output_ids == [10, 11]
    assert req.kv_allocated_len == req.kv_committed_len == 4

    scheduler.run_until_complete()
    assert req.output_ids == [10, 11, 12, 13]
    assert req.status is RequestStatus.FINISHED
    assert scheduler.result_queue == ()
    assert scheduler.future_map == ()
    assert scheduler.req_pool.active_count == 0
    assert scheduler.kv_pool.active_count == 0


def test_pipeline_new_prefill_breaks_decode_chain_before_launching_more_decode():
    """教学 barrier：waiting prefill 到来时，停止继续扩张 decode chain。"""
    runner = FakeLazyRunner(
        prefill_tokens=[10, 20], decode_tokens=[11, 12, 13, 21]
    )
    scheduler = MiniOverlapScheduler(runner, max_running_reqs=3, max_total_tokens=12)
    first = make_req("first", max_new_tokens=4)
    scheduler.add_request(first)
    scheduler.pipeline_step()
    scheduler.pipeline_step()
    scheduler.pipeline_step()
    assert scheduler.result_queue[0].kind == "chained"

    second = make_req("second", ids=[7], max_new_tokens=2)
    scheduler.add_request(second)
    turn = scheduler.pipeline_step()

    assert turn.barrier_reason == "prefill_waiting"
    assert [batch.mode for batch in turn.launched_batches] == [ForwardMode.EXTEND]
    assert turn.queue[0].rids == ("second",)
    second_prefill = runner.events.index("prefill_start:second")
    previous_finalize = max(
        i for i, event in enumerate(runner.events[:second_prefill])
        if event == "decode_finalize:first"
    )
    assert previous_finalize < second_prefill


def test_pipeline_drops_already_launched_token_and_defers_physical_release():
    """教学所有权：B0 完成请求时，B1 可能已经持有该请求。"""
    runner = FakeLazyRunner(
        prefill_tokens=[10, 20], decode_tokens=[11, 21, 12, 22, 23]
    )
    scheduler = MiniOverlapScheduler(runner, max_running_reqs=3, max_total_tokens=16)
    short = make_req("short", max_new_tokens=2)
    long = make_req("long", max_new_tokens=4)
    scheduler.add_request(short)
    scheduler.add_request(long)

    scheduler.pipeline_step()
    scheduler.pipeline_step()
    # B1 在 short 的 B0 结果返回前已经启动，所以它会多算一个 short token。
    # scheduler 必须丢弃该 token，并等 B1 离开队列后再 remove short。
    turn = scheduler.pipeline_step()

    assert turn.barrier_reason == "request_finished"
    assert len(turn.processed_results) == 2
    assert short.output_ids == [10, 11]
    assert long.output_ids == [20, 21, 22]
    assert short.status is RequestStatus.FINISHED
    assert runner.events.index("decode_finalize:short,long") < runner.events.index(
        "remove:short"
    )
    assert runner.events.count("decode_finalize:short,long") == 2
    assert "short" not in turn.queue[0].rids
    assert turn.queue[0].rids == ("long",)


def test_manual_and_pipeline_drivers_cannot_be_mixed():
    scheduler = MiniOverlapScheduler(FakeLazyRunner(), max_total_tokens=8)
    scheduler.add_request(make_req())
    scheduler.pipeline_step()
    with pytest.raises(RuntimeError, match="cannot mix"):
        scheduler.finalize_pending()


def test_pipeline_finalize_failure_discards_dependencies_and_requeues_request():
    """教学恢复：B0 finalize 失败时，连同依赖它的 B1 一起撤销。"""
    runner = FakeLazyRunner(
        prefill_tokens=[10],
        decode_tokens=[11, 12],
        fail_decode_finalize_at=1,
    )
    scheduler = MiniOverlapScheduler(runner, max_total_tokens=8)
    req = make_req(max_new_tokens=4)
    scheduler.add_request(req)
    scheduler.pipeline_step()
    scheduler.pipeline_step()

    # 失败发生时队列中同时存在 B0/B1；恢复后不能留下 future、row 或 KV，
    # 但 prefill 已经确认的 output token=10 必须保留用于 replay。
    with pytest.raises(RuntimeError, match="injected decode finalize failure"):
        scheduler.pipeline_step()

    assert scheduler.result_queue == ()
    assert scheduler.future_map == ()
    assert req.status is RequestStatus.WAITING
    assert req.output_ids == [10]
    assert req.req_pool_idx is None
    assert scheduler.req_pool.active_count == 0
    assert scheduler.kv_pool.active_count == 0
    assert runner.events.count("discard:r0") == 2


def test_pipeline_drain_processes_inflight_job_without_launching_another():
    runner = FakeLazyRunner(prefill_tokens=[10], decode_tokens=[11])
    scheduler = MiniOverlapScheduler(runner, max_total_tokens=8)
    req = make_req(max_new_tokens=3)
    scheduler.add_request(req)
    scheduler.pipeline_step()

    results = scheduler.drain()

    assert len(results) == 1
    assert results[0].batch is not None
    assert results[0].batch.mode is ForwardMode.EXTEND
    assert req.output_ids == [10]
    assert req.status is RequestStatus.RUNNING
    assert scheduler.result_queue == ()
    assert not any(event.startswith("decode_start") for event in runner.events)
