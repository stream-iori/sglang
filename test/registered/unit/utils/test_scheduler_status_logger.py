import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from sglang.srt.utils.scheduler_status_logger import SchedulerStatusLogger
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1)


class TestSchedulerStatusLoggerUnit(CustomTestCase):
    def test_dump_includes_batch_pool_and_cache_summaries(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = SchedulerStatusLogger(targets=[temp_dir], dump_interval=0)
            req = _req("run-1", req_pool_idx=3)
            queued = _req("wait-1")
            batch = SimpleNamespace(
                reqs=[req],
                forward_mode=SimpleNamespace(name="DECODE"),
                global_forward_mode=None,
                forward_iter=7,
                batch_is_full=False,
                is_prefill_only=False,
                seq_lens_cpu=[12],
                req_pool_indices_cpu=[3],
                prefix_lens=[8],
                extend_lens=[1],
                extend_num_tokens=1,
                seq_lens_sum=12,
                out_cache_loc=[42],
                req_to_token_pool=FakeReqToTokenPool(),
                token_to_kv_pool_allocator=FakeTokenToKVPoolAllocator(),
                tree_cache=FakeRadixCache(),
            )
            scheduler = SimpleNamespace(
                req_to_token_pool=batch.req_to_token_pool,
                token_to_kv_pool_allocator=batch.token_to_kv_pool_allocator,
                tree_cache=batch.tree_cache,
                cur_batch=batch,
                last_batch=None,
            )

            logger.maybe_dump(batch, [queued], scheduler)

            data = _read_first_event(temp_dir)
            self.assertEqual(data["event"], "scheduler.status")
            self.assertEqual(data["running_rids"], ["run-1"])
            self.assertEqual(data["queued_rids"], ["wait-1"])
            self.assertEqual(data["running_batch"]["forward_mode"], "DECODE")
            self.assertEqual(data["running_batch"]["req_pool_indices"], [3])
            self.assertEqual(data["cur_batch"]["forward_mode"], "DECODE")
            self.assertEqual(data["cur_batch"]["req_pool_indices"], [3])
            self.assertIsNone(data["last_batch"])
            self.assertEqual(data["waiting_queue"]["size"], 1)
            self.assertEqual(data["req_to_token_pool"]["class"], "FakeReqToTokenPool")
            self.assertEqual(data["req_to_token_pool"]["alloc_size"], 11)
            self.assertEqual(data["req_to_token_pool"]["used_size"], 6)
            self.assertEqual(data["req_to_token_pool"]["free_slots_head"], [4, 5, 6, 7])
            self.assertEqual(
                data["req_to_token_pool"]["active_rows"][0]["token_locs_head"],
                [300, 301, 302, 303, 304, 305, 306, 307],
            )
            self.assertEqual(
                data["req_to_token_pool"]["active_rows"][0]["token_locs_tail"],
                [304, 305, 306, 307, 308, 309, 310, 311],
            )
            self.assertEqual(
                data["token_to_kv_pool_allocator"]["available_size"], 64
            )
            self.assertEqual(data["radix_cache"]["class"], "FakeRadixCache")
            self.assertEqual(data["radix_cache"]["total_size"], 11)
            self.assertEqual(data["radix_cache"]["root_children"], 2)


def _req(rid, req_pool_idx=None):
    return SimpleNamespace(
        rid=rid,
        req_pool_idx=req_pool_idx,
        origin_input_ids=[1, 2, 3],
        output_ids=[4],
        prefix_indices=[1, 2],
        extend_input_len=1,
        priority=None,
        finished_reason=None,
        sampling_params=SimpleNamespace(max_new_tokens=8),
    )


class FakeReqToTokenPool:
    size = 10
    _alloc_size = 11
    max_context_len = 128
    device = "cpu"
    free_slots = [4, 5, 6, 7]
    req_to_token = None

    def __init__(self):
        self.req_to_token = FakeReqToTokenTable()

    def available_size(self):
        return len(self.free_slots)


class FakeReqToTokenTable:
    shape = (11, 128)

    def __getitem__(self, key):
        row_idx, col_slice = key
        row = [row_idx * 100 + i for i in range(self.shape[1])]
        return row[col_slice]


class FakeTokenToKVPoolAllocator:
    size = 100
    page_size = 1

    def available_size(self):
        return 64


class FakeRadixCache:
    disable = False
    page_size = 1
    root_node = SimpleNamespace(children={"a": object(), "b": object()})

    def total_size(self):
        return 11

    def evictable_size(self):
        return 5

    def protected_size(self):
        return 6

    def full_evictable_size(self):
        return 5

    def swa_evictable_size(self):
        return 0

    def is_tree_cache(self):
        return True

    def is_chunk_cache(self):
        return False

    def supports_mamba(self):
        return False


def _read_first_event(log_dir):
    log_file = next(Path(log_dir).glob("*.log"))
    for line in log_file.read_text().splitlines():
        idx = line.find("{")
        if idx != -1:
            return json.loads(line[idx:])
    raise AssertionError("no JSON event found")


if __name__ == "__main__":
    unittest.main()
