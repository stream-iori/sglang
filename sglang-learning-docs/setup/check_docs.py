#!/usr/bin/env python3
"""Check local documentation links and the current source reading contracts.

Run from any directory. Uses only Python's standard library; does not import SRT.
"""

import ast
import re
from pathlib import Path
from urllib.parse import unquote

DOCS = Path(__file__).resolve().parents[1]
ROOT = DOCS.parent

CONTRACTS = {
    "python/sglang/srt/managers/scheduler.py": [
        "Scheduler.event_loop_normal", "Scheduler.event_loop_overlap",
        "Scheduler.get_next_batch_to_run", "Scheduler.run_batch",
        "Scheduler.process_batch_result", "Scheduler.init_model_worker",
    ],
    "python/sglang/srt/managers/schedule_batch.py": [
        "Req", "NextBatchPlan", "ScheduleBatch.prepare_for_extend",
        "ScheduleBatch.prepare_for_decode", "ScheduleBatch.filter_batch",
        "ScheduleBatch.merge_batch", "ScheduleBatch.is_empty",
    ],
    "python/sglang/srt/model_executor/forward_batch_info.py": [
        "ForwardMode", "ForwardBatch.init_new",
    ],
    "python/sglang/srt/model_executor/model_runner.py": [
        "ModelRunner.forward", "ModelRunner._forward_raw",
    ],
    "python/sglang/srt/model_executor/runner/eager_runner.py": ["EagerRunner"],
    "python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py": [
        "DecodeCudaGraphRunner",
    ],
    "python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py": [
        "PrefillCudaGraphRunner",
    ],
    "python/sglang/srt/mem_cache/registry.py": [
        "default_radix_cache_factory", "create_unified_radix_cache",
    ],
    "python/sglang/srt/mem_cache/unified_radix_cache.py": ["UnifiedRadixCache"],
    "python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py": [
        "UnifiedTreeCore", "UnifiedTreeNode",
    ],
    "python/sglang/srt/mem_cache/unified_cache/components/full.py": ["FullComponent"],
    "python/sglang/srt/mem_cache/unified_cache/components/swa.py": ["SWAComponent"],
    "python/sglang/srt/mem_cache/unified_cache/components/mamba.py": ["MambaComponent"],
    "python/sglang/srt/mem_cache/kv_loc_plan.py": [
        "IdSpaceKind", "IdSpace", "KVLocPlan.bind",
        "KVLocPlan.read_table", "KVLocPlan.write_ids",
    ],
    "python/sglang/srt/mem_cache/allocation.py": [
        "alloc_for_extend", "alloc_for_decode", "assign_req_to_token_pool",
    ],
    "python/sglang/srt/mem_cache/prefill_budget.py": [
        "PrefillBudget", "SWAPrefillBudget", "SharedSWAPrefillBudget",
    ],
    "python/sglang/srt/runtime_context.py": [
        "RuntimeContext.override", "publish", "get_exec", "get_memory",
        "get_flags", "get_resources", "get_forward",
    ],
    "python/sglang/srt/disaggregation/decode.py": [
        "DecodePreallocQueue", "DecodeTransferQueue",
    ],
    "python/sglang/srt/observability/req_time_stats.py": ["RequestStage"],
    "python/sglang/srt/hardware_backend/mps/runtime.py": ["validate_mps_runtime"],
}


def collect_symbols(node, prefix=""):
    result = set()
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            name = f"{prefix}.{child.name}" if prefix else child.name
            result.add(name)
            result.update(collect_symbols(child, name))
        else:
            result.update(collect_symbols(child, prefix))
    return result


def main():
    errors = []
    link_count = 0
    pages = sorted(DOCS.rglob("*.md"))
    for page in pages:
        content = page.read_text()
        # Ignore examples inside fenced blocks when looking for markdown links.
        prose = re.sub(r"^```.*?^```[^\n]*$", "", content, flags=re.M | re.S)
        for match in re.finditer(r"\[[^\]\n]+\]\(([^)]+)\)", prose):
            target = match.group(1).strip()
            if target.startswith(("https://", "http://", "mailto:", "#")):
                continue
            target = unquote(target.split("#", 1)[0].strip("<>"))
            if not target:
                continue
            link_count += 1
            if not (page.parent / target).exists():
                errors.append(f"{page.relative_to(DOCS)}: missing link {target}")
        if "b8d7351a74" in content:
            errors.append(f"{page.relative_to(DOCS)}: obsolete source baseline")
        fences = re.findall(r"^```", content, flags=re.M)
        if len(fences) % 2:
            errors.append(f"{page.relative_to(DOCS)}: unclosed code fence")

    symbol_count = 0
    for filename, names in CONTRACTS.items():
        path = ROOT / filename
        if not path.exists():
            errors.append(f"missing source file: {filename}")
            continue
        found = collect_symbols(ast.parse(path.read_text()))
        for name in names:
            symbol_count += 1
            if name not in found:
                errors.append(f"{filename}: missing symbol {name}")

    for error in errors:
        print(f"ERROR: {error}")
    print(f"Checked {len(pages)} pages, {link_count} local links, "
          f"{symbol_count} source symbols; {len(errors)} errors")
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
