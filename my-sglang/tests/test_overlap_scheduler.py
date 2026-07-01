from __future__ import annotations

from my_sglang.models import Req, RequestStatus, SamplingParams
from my_sglang.overlap_scheduler import MiniOverlapScheduler


class FakeLazyRunner:
    # FakeLazyRunner 模拟 MLX lazy API：start 返回 handle，kick 记录“已提交”，finalize 才返回 token。
    def __init__(self, prefill_tokens=None, decode_tokens=None):
        self.prefill_tokens = list(prefill_tokens or [100])
        self.decode_tokens = list(decode_tokens or [101, 102, 103])
        self.events: list[str] = []
        self.removed: list[str] = []
        self.live: set[str] = set()

    def prefill(
        self,
        req_id,
        new_token_ids,
        full_token_ids,
        prefix_slot_ids,
        new_slot_ids,
        req_pool_idx,
    ):
        handle = self.prefill_start(
            req_id,
            new_token_ids,
            full_token_ids,
            prefix_slot_ids,
            new_slot_ids,
            req_pool_idx,
        )
        self.prefill_kick(handle)
        return self.prefill_finalize(handle)

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
        self.live.add(req_id)
        return {"kind": "prefill", "rid": req_id, "token": self.prefill_tokens.pop(0)}

    def prefill_kick(self, pending):
        self.events.append(f"prefill_kick:{pending['rid']}")

    def prefill_finalize(self, pending):
        self.events.append(f"prefill_finalize:{pending['rid']}")
        return pending["token"]

    def decode_batch_start(self, req_ids):
        rid_key = ",".join(req_ids)
        self.events.append(f"decode_start:{rid_key}")
        return {
            "kind": "decode",
            "rids": list(req_ids),
            "tokens": [self.decode_tokens.pop(0) for _ in req_ids],
        }

    def decode_batch_kick(self, pending):
        self.events.append(f"decode_kick:{','.join(pending['rids'])}")

    def decode_batch_finalize(self, pending):
        self.events.append(f"decode_finalize:{','.join(pending['rids'])}")
        return pending["tokens"]

    def remove_request(self, req_id):
        self.events.append(f"remove:{req_id}")
        self.removed.append(req_id)
        self.live.discard(req_id)


def make_req(rid="r0", ids=None, max_new_tokens=2):
    # overlap 测试只关心长度停止，不引入 eos 的不确定性。
    return Req(
        rid=rid,
        origin_input_ids=[1, 2] if ids is None else list(ids),
        sampling_params=SamplingParams(
            max_new_tokens=max_new_tokens,
            eos_token_ids=frozenset(),
        ),
    )


def test_overlap_scheduler_uses_start_kick_finalize_for_prefill_and_decode():
    runner = FakeLazyRunner(prefill_tokens=[10], decode_tokens=[11])
    scheduler = MiniOverlapScheduler(runner, max_running_reqs=2, max_total_tokens=8)
    req = make_req(max_new_tokens=2)

    scheduler.add_request(req)
    first = scheduler.step()
    second = scheduler.step()

    assert first.prefill_batch is not None
    assert first.decode_batch is None
    assert second.decode_batch is not None
    assert req.status is RequestStatus.FINISHED
    assert req.output_ids == [10, 11]
    assert runner.events == [
        "prefill_start:r0",
        "prefill_kick:r0",
        "prefill_finalize:r0",
        "decode_start:r0",
        "decode_kick:r0",
        "decode_finalize:r0",
        "remove:r0",
    ]


def test_overlap_launches_new_prefill_and_old_decode_before_finalize():
    runner = FakeLazyRunner(prefill_tokens=[10, 20], decode_tokens=[11])
    scheduler = MiniOverlapScheduler(runner, max_running_reqs=4, max_total_tokens=16)
    old_req = make_req("old", [1], max_new_tokens=2)
    new_req = make_req("new", [2], max_new_tokens=2)

    scheduler.add_request(old_req)
    scheduler.step()
    runner.events.clear()

    scheduler.add_request(new_req)
    result = scheduler.step()

    assert result.prefill_batch is not None
    assert result.decode_batch is not None
    assert [req.rid for req in result.decode_batch.reqs] == ["old"]
    assert new_req.output_ids == [20]
    assert old_req.status is RequestStatus.FINISHED
    assert runner.events[:4] == [
        "prefill_start:new",
        "prefill_kick:new",
        "decode_start:old",
        "decode_kick:old",
    ]
    assert "decode_finalize:old" in runner.events
    assert "prefill_finalize:new" in runner.events
