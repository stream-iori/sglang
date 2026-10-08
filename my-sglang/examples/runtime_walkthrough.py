"""Deterministic documentation experiments; run from my-sglang with uv run."""

import argparse
import json
from dataclasses import asdict

from my_sglang.models import Req, RequestStatus, SamplingParams
from my_sglang.overlap_scheduler import MiniOverlapScheduler
from my_sglang.runner import FakeCudaRunner
from my_sglang.scheduler import MiniScheduler


class ConfirmedCountRunner:
    """Produces synthetic tokens from confirmed history, including after retract.

    Intermediate chunks produce a value that the scheduler must ignore.
    This runner does not compute K/V vectors or model logits.
    """

    def run_batch(self, forward, req_to_token_pool):
        return [
            {"A": 40, "B": 50, "C": 60}[req.rid] + len(req.output_ids)
            for req in forward.reqs
        ]

    def remove_request(self, rid):
        pass


def request(rid, prompt, count):
    return Req(rid=rid, origin_input_ids=prompt, sampling_params=SamplingParams(count))


def report(step, scheduler, reqs, batches, **extra):
    cache = scheduler.tree_cache
    pages = []
    if cache:
        def visit(node):
            if node.value:
                pages.append({"tokens": node.key, "slots": node.value, "lock": node.lock_ref})
            for child in node.children.values():
                visit(child)
        visit(cache.root_node)
    print(json.dumps({
        "step": step,
        "batches": [{"mode": f.forward_mode.value,
                     "rids": [r.rid for r in f.reqs],
                     "inputs": f.input_ids_by_req,
                     "slots": f.out_cache_loc_by_req} for f in batches],
        "reqs": [{"rid": r.rid, "status": r.status.value, "output": r.output_ids,
                  "row": r.req_pool_idx, "allocated": r.kv.kv_allocated_len,
                  "committed": r.kv_committed_len,
                  "mapping": ([] if r.req_pool_idx is None else
                              scheduler.req_to_token_pool.row(r.req_pool_idx, r.kv.kv_allocated_len).tolist())}
                 for r in reqs],
        "memory": asdict(scheduler.memory_snapshot()), "cache": pages, **extra,
    }, ensure_ascii=False))


def single(overlap=False):
    runner = FakeCudaRunner(tokens=[10, 11, 12, 13])
    cls = MiniOverlapScheduler if overlap else MiniScheduler
    scheduler = cls(runner, max_total_tokens=8, page_size=2, trace=False)
    req = request("A", [7, 8], 3)
    scheduler.add_request(req)
    for step in range(1, 12):
        if overlap:
            turn = scheduler.pipeline_step()
            report(step, scheduler, [req], turn.launched_batches,
                   processed=[x.batch.forward_mode.value for x in turn.processed_results if x.batch],
                   future=scheduler.future_map.snapshot(),
                   queue=[asdict(x) for x in turn.queue])
        else:
            turn = scheduler.step()
            report(step, scheduler, [req], [turn.batch] if turn.batch else [])
        if req.status is RequestStatus.FINISHED:
            assert req.output_ids == [10, 11, 12]
            assert scheduler.req_to_token_pool.active_count == 0
            print(json.dumps({"trace": runner.trace}))
            return
    raise AssertionError("single case did not finish")


def shared():
    scheduler = MiniScheduler(ConfirmedCountRunner(), max_total_tokens=8,
                              page_size=2, max_running_reqs=3, new_token_ratio=0,
                              chunked_prefill_size=2, enable_radix_cache=True, trace=False)
    a, b = request("A", [7, 8, 9, 10], 3), request("B", [7, 8, 11], 3)
    c = request("C", [20, 21, 22, 23, 24, 25], 1)
    active = [a]
    scheduler.add_request(a)
    retracted = []
    added_c = False
    for step in range(1, 31):
        if step == 2:
            scheduler.add_request(b)
            active.append(b)
        if not added_c and a.finished_reason and b.finished_reason:
            scheduler.add_request(c)
            active.append(c)
            added_c = True
        turn = scheduler.step()
        retracted.extend(turn.retracted_rids)
        scheduler.assert_consistent()
        report(step, scheduler, active, [turn.batch] if turn.batch else [],
               retract=turn.retracted_rids, abort=turn.aborted_rids)
        assert not turn.aborted_rids
        if c.finished_reason:
            assert retracted == ["A"]
            assert a.output_ids == [40, 41, 42] and b.output_ids == [50, 51, 52]
            assert c.output_ids == [60]
            assert scheduler.tree_cache.match_prefix([7, 8, 9, 10], pin=False).token_count == 2
            assert scheduler.tree_cache.match_prefix(c.origin_input_ids, pin=False).token_count == 6
            return
    raise AssertionError("shared case did not finish")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=["sync", "overlap", "shared"])
    args = parser.parse_args()
    if args.case == "shared":
        shared()
    else:
        single(overlap=args.case == "overlap")
