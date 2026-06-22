import unittest

from sglang.srt.observability.req_time_stats import (
    RequestStage,
    trace_request_event,
    trace_request_slice,
    trace_request_stage,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeTraceContext:
    def __init__(self, tracing_enable: bool):
        self.tracing_enable = tracing_enable
        self.events = []

    def trace_event(self, event_name, level, timestamp_ns, attrs=None):
        self.events.append((event_name, level, timestamp_ns, attrs))


class _FakeTimeStats:
    def __init__(self, tracing_enable: bool = True):
        self.trace_ctx = _FakeTraceContext(tracing_enable)
        self.slices = []

    def trace_slice(self, stage, start_time, end_time, attrs=None):
        self.slices.append((stage.stage_name, start_time, end_time, attrs))


class _FakeReq:
    def __init__(self, tracing_enable: bool = True):
        self.time_stats = _FakeTimeStats(tracing_enable)


class TestMlxTraceHelpers(CustomTestCase):
    def test_trace_request_slice_filters_disabled_requests(self):
        enabled_req = _FakeReq(True)
        disabled_req = _FakeReq(False)

        trace_request_slice(
            [enabled_req, disabled_req],
            RequestStage.MLX_PREFILL,
            1.0,
            2.0,
            {"backend": "mlx"},
        )

        self.assertEqual(len(enabled_req.time_stats.slices), 1)
        self.assertEqual(enabled_req.time_stats.slices[0][0], "mlx_prefill")
        self.assertEqual(enabled_req.time_stats.slices[0][3], {"backend": "mlx"})
        self.assertEqual(disabled_req.time_stats.slices, [])

    def test_trace_request_event_supports_single_req(self):
        req = _FakeReq(True)

        trace_request_event(req, "mlx.overlap.launch_fresh", {"batch_size": 1})

        self.assertEqual(len(req.time_stats.trace_ctx.events), 1)
        event_name, level, timestamp_ns, attrs = req.time_stats.trace_ctx.events[0]
        self.assertEqual(event_name, "mlx.overlap.launch_fresh")
        self.assertEqual(level, 3)
        self.assertGreater(timestamp_ns, 0)
        self.assertEqual(attrs, {"batch_size": 1})

    def test_trace_request_stage_records_slice_on_exception(self):
        req = _FakeReq(True)

        with self.assertRaises(RuntimeError):
            with trace_request_stage(req, RequestStage.MLX_DECODE, {"mode": "decode"}):
                raise RuntimeError("boom")

        self.assertEqual(len(req.time_stats.slices), 1)
        stage_name, start_time, end_time, attrs = req.time_stats.slices[0]
        self.assertEqual(stage_name, "mlx_decode")
        self.assertLessEqual(start_time, end_time)
        self.assertEqual(attrs, {"mode": "decode"})


if __name__ == "__main__":
    unittest.main()
