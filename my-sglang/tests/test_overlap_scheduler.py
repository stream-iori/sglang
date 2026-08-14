from __future__ import annotations

import numpy as np
import pytest

from my_sglang.models import ForwardMode, Req, RequestStatus, SamplingParams
from my_sglang.overlap_scheduler import FutureMap, MiniOverlapScheduler
from my_sglang.runner import FakeCudaRunner


def make_req(rid="r0", max_new_tokens=4) -> Req:
    return Req(rid=rid, origin_input_ids=[1, 2], sampling_params=SamplingParams(max_new_tokens))


def test_future_map_is_pool_indexed_device_token_buffer():
    future = FutureMap(4)
    future.stash(np.asarray([3, 1]), np.asarray([101, 205]))
    assert future.gather(np.asarray([1, 3])).tolist() == [205, 101]
    with pytest.raises(RuntimeError, match="before producer"):
        future.gather(np.asarray([1]))
    future.clear(3)
    with pytest.raises(RuntimeError, match="before producer"):
        future.gather(np.asarray([3]))


def test_fake_cuda_event_only_advances_required_forward_prefix():
    runner = FakeCudaRunner(tokens=[10, 11])
    future = FutureMap(2)
    first = runner.run_batch_async(_forward(ForwardMode.EXTEND, (0,)), future)
    second = runner.run_batch_async(_forward(ForwardMode.DECODE, (0,)), future)
    assert runner.resolve(first) == [10]
    assert future.snapshot() == ((0, 10),)
    assert not second.copy_done.ready
    assert runner.resolve(second) == [11]


def _forward(mode: ForwardMode, rows: tuple[int, ...]):
    from my_sglang.models import ForwardBatch
    reqs = tuple(make_req(f"r{i}") for i in range(len(rows)))
    return ForwardBatch(
        mode,
        reqs,
        tuple(1 for _ in rows),
        rows,
        tuple(0 for _ in rows),
        tuple(1 for _ in rows),
        extend_seq_lens=tuple(1 for _ in rows),
    )


def test_pipeline_launches_b1_before_waiting_b0_copy_event():
    runner = FakeCudaRunner(tokens=[10, 11, 12, 13])
    scheduler = MiniOverlapScheduler(runner, max_running_reqs=2, max_total_tokens=8)
    req = make_req()
    scheduler.add_request(req)
    scheduler.pipeline_step()  # prefill B0
    scheduler.pipeline_step()  # process prefill, launch decode B1
    runner.trace.clear()
    turn = scheduler.pipeline_step()  # enqueue decode B2, then process B1
    enqueue_b2 = runner.trace.index("forward:enqueue:forward+sample:B2")
    sync_b1 = runner.trace.index("event:sync:B1.copy_done")
    assert enqueue_b2 < sync_b1
    assert turn.queue_depth == 1
    assert req.output_ids == [10, 11]
    assert "forward:gather:B1:[10]" in runner.trace


def test_pipeline_discards_speculative_extra_and_defers_release():
    runner = FakeCudaRunner(tokens=[10, 11, 12])
    scheduler = MiniOverlapScheduler(runner, max_total_tokens=8)
    req = make_req(max_new_tokens=2)
    scheduler.add_request(req)
    scheduler.pipeline_step(); scheduler.pipeline_step(); scheduler.pipeline_step()
    assert req.output_ids == [10, 11]
    assert req.status is RequestStatus.FINISHED
    assert scheduler.req_to_token_pool.active_count == 0
    assert any(item.startswith("pipeline_drop_token") for item in []) is False


def test_pipeline_failure_clears_relay_and_requeues_unconfirmed_request():
    runner = FakeCudaRunner(tokens=[10, 11, 12], fail_resolve_at=2)
    scheduler = MiniOverlapScheduler(runner, max_total_tokens=8)
    req = make_req()
    scheduler.add_request(req)
    scheduler.pipeline_step()
    scheduler.pipeline_step()
    with pytest.raises(RuntimeError, match="resolve failure"):
        scheduler.pipeline_step()
    assert not scheduler.result_queue
    assert scheduler.future_map.snapshot() == ()
    assert req.status is RequestStatus.WAITING
    assert req.output_ids == [10]


def test_pipeline_chunked_prefill_clears_chunk_owner_before_decode():
    # B4 is submitted before B3 reports the length stop; its token is discarded.
    runner = FakeCudaRunner(tokens=[90, 91, 10, 11, 12])
    scheduler = MiniOverlapScheduler(
        runner,
        max_total_tokens=8,
        chunked_prefill_size=2,
    )
    req = Req(
        rid="chunked",
        origin_input_ids=[1, 2, 3, 4, 5],
        sampling_params=SamplingParams(max_new_tokens=2),
    )
    scheduler.add_request(req)

    completed = scheduler.run_until_complete()

    assert completed == [req]
    assert req.output_ids == [10, 11]
    assert req.status is RequestStatus.FINISHED
    assert scheduler.chunked_req is None
    assert not scheduler.result_queue
    assert scheduler.future_map.snapshot() == ()
    assert scheduler.req_to_token_pool.active_count == 0
    assert scheduler.token_to_kv_pool_allocator.allocated_size == 0
