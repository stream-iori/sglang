from __future__ import annotations

from pathlib import Path

import pytest

from my_sglang.models import Req, RequestStatus, SamplingParams
from my_sglang.runner import SglangMlxRunnerAdapter
from my_sglang.scheduler import MiniScheduler


MODEL_PATH = Path.home() / ".modelscope/models/Qwen3-0.6B"


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="本地缺少 ModelScope Qwen3-0.6B")
def test_real_mlx_prefill_decode_lifecycle():
    # 真实集成测试：复用 SGLang 的 MlxModelRunner，确认 mini scheduler 能跑通实际模型。
    runner = SglangMlxRunnerAdapter(str(MODEL_PATH), mem_fraction_static=0.2)
    scheduler = MiniScheduler(runner, max_running_reqs=2, max_total_tokens=16)
    req = Req(
        rid="mlx-r0",
        origin_input_ids=[9707],
        sampling_params=SamplingParams(max_new_tokens=2, eos_token_ids=frozenset()),
    )

    scheduler.add_request(req)
    scheduler.run_until_complete()

    assert req.status is RequestStatus.FINISHED
    assert len(req.output_ids) == 2
    assert scheduler.last_prefill_batch is not None
    assert scheduler.last_decode_batch is not None
    assert scheduler.req_pool.active_count == 0
    assert scheduler.kv_pool.active_count == 0
    assert not runner.has_request("mlx-r0")
