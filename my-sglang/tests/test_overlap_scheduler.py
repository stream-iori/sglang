from __future__ import annotations

import pytest
from typing import Any

from my_sglang.models import ForwardMode, Req, RequestStatus, SamplingParams
from my_sglang.overlap_scheduler import MiniOverlapScheduler


class FakeLazyRunner:
    """Records the explicit start -> kick -> finalize boundary."""

    def __init__(self, prefill_tokens=None, decode_tokens=None, extend_tokens=None):
        self.prefill_tokens = list(prefill_tokens or [100, 110])
        self.decode_tokens = list(decode_tokens or [101, 102, 103])
        self.extend_tokens = list(extend_tokens or [201, 202, 203])
        self.events: list[str] = []

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

    def decode_batch_kick(self, pending: Any) -> None:
        self.events.append(f"decode_kick:{','.join(pending['rids'])}")

    def decode_batch_finalize(self, pending: Any) -> list[int]:
        self.events.append(f"decode_finalize:{','.join(pending['rids'])}")
        return pending["tokens"]

    def remove_request(self, req_id):
        self.events.append(f"remove:{req_id}")


def make_req(rid="r0", ids=None, max_new_tokens=2):
    return Req(
        rid=rid,
        origin_input_ids=[1, 2] if ids is None else list(ids),
        sampling_params=SamplingParams(max_new_tokens=max_new_tokens),
    )


def test_launch_allocates_and_finalize_commits_prefill_and_decode():
    runner = FakeLazyRunner(prefill_tokens=[10], decode_tokens=[11])
    scheduler = MiniOverlapScheduler(runner, max_running_reqs=2, max_total_tokens=8)
    req = make_req()
    scheduler.add_request(req)

    launched = scheduler.launch_step()
    assert launched.batch is not None
    assert launched.batch.mode is ForwardMode.EXTEND
    assert req.kv_allocated_len == 2
    assert req.kv_committed_len == 0
    assert req.output_ids == []
    assert runner.events == ["prefill_start:r0", "prefill_kick:r0"]

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
