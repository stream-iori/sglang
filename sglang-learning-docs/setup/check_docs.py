#!/usr/bin/env python3
"""Check local documentation links and the current source reading contracts.

Run from any directory. Uses only Python's standard library; does not import SRT.
"""

import ast
import hashlib
import json
import subprocess
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
        "get_flags", "get_resources", "get_forward", "create_context", "use_context",
    ],
    "python/sglang/srt/mem_cache/device_pool_info.py": [
        "DevicePoolInfo", "MHABufferInfo", "MLABufferInfo", "MambaStateBufferInfo",
    ],
    "python/sglang/srt/model_executor/runner/row_bucket_graph_runner.py": [
        "RowBucketGraphRunner.ready", "RowBucketGraphRunner.run",
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


def heading_anchors(content):
    anchors = set()
    counts = {}
    for heading in re.findall(r"^#{1,6} +(.+)$", strip_fences(content), re.M):
        heading = re.sub(r"<[^>]+>", "", heading).strip().lower()
        slug = re.sub(r"[^\w\- ]", "", heading).replace(" ", "-")
        count = counts.get(slug, 0)
        counts[slug] = count + 1
        anchors.add(slug if not count else f"{slug}-{count}")
    return anchors


def strip_fences(content):
    return re.sub(r"^```[^\n]*\n.*?^```[^\n]*$", "", content, flags=re.M | re.S)


def source_digest(directory):
    """Hash tracked files and relative names; ignored virtualenv/cache files stay out."""
    names = subprocess.check_output(
        ["git", "ls-files", "-z", "--", directory], cwd=ROOT
    ).decode().split("\0")
    digest = hashlib.sha256()
    for name in sorted(n for n in names if n):
        path = ROOT / name
        digest.update(name.encode() + b"\0")
        digest.update(path.read_bytes() if path.is_file() else b"<MISSING>")
        digest.update(b"\0")
    return digest.hexdigest()


def main():
    errors = []
    link_count = block_count = 0
    pages = sorted(DOCS.rglob("*.md"))
    baseline = json.loads((DOCS / "setup/source-baseline.json").read_text())
    audit = json.loads((DOCS / "setup/doc-audit.json").read_text())
    expected_pages = {str(p.relative_to(DOCS)) for p in pages}
    if set(audit) != expected_pages:
        errors.append("doc-audit.json does not cover the exact current page inventory")
    for page in pages:
        content = page.read_text()
        relative = str(page.relative_to(DOCS))
        if page.name != "mac-validation.md" and baseline["upstream_commit"][:10] not in content:
            errors.append(f"{relative}: missing current source baseline")
        entry = audit.get(relative, {})
        if not entry.get("evidence") or not entry.get("review"):
            errors.append(f"{relative}: missing review rationale/source evidence")
        for source in entry.get("evidence", []):
            if not (ROOT / source).exists():
                errors.append(f"{relative}: missing audit source {source}")
        prose = strip_fences(content)
        for match in re.finditer(r"\[[^\]\n]+\]\(([^)]+)\)", prose):
            target = match.group(1).strip().strip("<>")
            if target.startswith(("https://", "http://", "mailto:")):
                continue
            path_part, _, anchor = target.partition("#")
            target_path = page.parent / unquote(path_part) if path_part else page
            link_count += 1
            if not target_path.exists():
                errors.append(f"{relative}: missing link {target}")
            elif anchor and target_path.suffix == ".md":
                if unquote(anchor) not in heading_anchors(target_path.read_text()):
                    errors.append(f"{relative}: missing anchor {target}")
        if len(re.findall(r"^```", content, re.M)) % 2:
            errors.append(f"{relative}: unclosed code fence")
        for lang, code in re.findall(r"^```([^\n]*)\n(.*?)^```[^\n]*$", content, re.M | re.S):
            try:
                if lang.strip() in {"python", "py"}:
                    ast.parse(code)
                    block_count += 1
                elif lang.strip() in {"bash", "sh"}:
                    result = subprocess.run(["bash", "-n"], input=code, text=True, capture_output=True)
                    if result.returncode:
                        raise ValueError(result.stderr.strip())
                    block_count += 1
                elif lang.strip() == "json":
                    json.loads(code)
                    block_count += 1
            except (SyntaxError, ValueError) as exc:
                errors.append(f"{relative}: invalid {lang} block: {exc}")
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
    # Syntax, not execution: these checks must never launch a model or install packages.
    for path in sorted(DOCS.rglob("*.py")):
        ast.parse(path.read_text(), filename=str(path))
    for path in sorted((DOCS / "setup").glob("*.sh")):
        result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
        if result.returncode:
            errors.append(f"{path.name}: {result.stderr.strip()}")
    for directory, expected in baseline["source_digests"].items():
        if source_digest(directory) != expected:
            errors.append(f"source drift in {directory}: review semantics and refresh baseline evidence")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", baseline["upstream_commit"], "HEAD"], cwd=ROOT
    )
    if ancestor.returncode:
        errors.append("documented upstream commit is not integrated into HEAD")
    for error in errors:
        print(f"ERROR: {error}")
    print(f"Checked {len(pages)} pages, {link_count} local links/anchors, "
          f"{symbol_count} source symbols, {block_count} code blocks; {len(errors)} errors")
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
