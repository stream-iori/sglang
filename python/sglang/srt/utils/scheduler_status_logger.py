from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, List, Optional

import torch.distributed as dist

from sglang.srt.environ import envs
from sglang.srt.utils.log_utils import create_log_targets, log_json

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch


MAX_SUMMARY_ITEMS = 16
MAX_VECTOR_ITEMS = 64
MAX_TOKEN_LOC_SAMPLE_ITEMS = 8


class SchedulerStatusLogger:
    def __init__(self, targets: List[str], dump_interval: float):
        self.loggers = create_log_targets(targets=targets, name_prefix=__name__)
        self.dump_interval = dump_interval
        self.last_dump_time = 0.0
        self.rank = dist.get_rank() if dist.is_initialized() else 0

    @staticmethod
    def maybe_create(enable_metrics: bool) -> Optional["SchedulerStatusLogger"]:
        target = envs.SGLANG_LOG_SCHEDULER_STATUS_TARGET.get()
        if not target:
            return None

        if not enable_metrics:
            raise ValueError(
                "SGLANG_LOG_SCHEDULER_STATUS_TARGET is set but --enable-metrics "
                "is not active. Status dumps require --enable-metrics to work."
            )

        return SchedulerStatusLogger(
            targets=[t.strip() for t in target.split(",") if t.strip()],
            dump_interval=envs.SGLANG_LOG_SCHEDULER_STATUS_INTERVAL.get(),
        )

    def maybe_dump(
        self,
        running_batch: "ScheduleBatch",
        waiting_queue: List["Req"],
        scheduler: Optional[Any] = None,
    ) -> None:
        now = time.time()
        if now - self.last_dump_time < self.dump_interval:
            return

        self.last_dump_time = now
        req_to_token_pool = _first_not_none(
            getattr(scheduler, "req_to_token_pool", None),
            getattr(running_batch, "req_to_token_pool", None),
        )
        token_to_kv_pool_allocator = _first_not_none(
            getattr(scheduler, "token_to_kv_pool_allocator", None),
            getattr(running_batch, "token_to_kv_pool_allocator", None),
        )
        tree_cache = _first_not_none(
            getattr(scheduler, "tree_cache", None),
            getattr(running_batch, "tree_cache", None),
        )
        cur_batch = getattr(scheduler, "cur_batch", None)
        last_batch = getattr(scheduler, "last_batch", None)
        log_json(
            self.loggers,
            "scheduler.status",
            {
                "rank": self.rank,
                "running_rids": [r.rid for r in running_batch.reqs],
                "queued_rids": [r.rid for r in waiting_queue],
                "running_batch": _summarize_running_batch(running_batch),
                "cur_batch": _summarize_optional_batch(cur_batch),
                "last_batch": _summarize_optional_batch(last_batch),
                "waiting_queue": _summarize_waiting_queue(waiting_queue),
                "req_to_token_pool": _summarize_req_to_token_pool(
                    req_to_token_pool, running_batch
                ),
                "token_to_kv_pool_allocator": _summarize_token_to_kv_pool_allocator(
                    token_to_kv_pool_allocator
                ),
                "radix_cache": _summarize_prefix_cache(tree_cache),
            },
        )


def _summarize_optional_batch(batch: Optional["ScheduleBatch"]) -> Optional[dict]:
    if batch is None:
        return None
    return _summarize_running_batch(batch)


def _summarize_running_batch(batch: "ScheduleBatch") -> dict:
    reqs = getattr(batch, "reqs", [])
    return {
        "size": len(reqs),
        "rids": [r.rid for r in reqs[:MAX_SUMMARY_ITEMS]],
        "forward_mode": _enum_name(getattr(batch, "forward_mode", None)),
        "global_forward_mode": _enum_name(getattr(batch, "global_forward_mode", None)),
        "forward_iter": _json_value(getattr(batch, "forward_iter", None)),
        "batch_is_full": _json_value(getattr(batch, "batch_is_full", None)),
        "is_prefill_only": _json_value(getattr(batch, "is_prefill_only", None)),
        "seq_lens": _to_int_list(
            _first_not_none(
                getattr(batch, "seq_lens_cpu", None), getattr(batch, "seq_lens", None)
            )
        ),
        "req_pool_indices": _to_int_list(
            _first_not_none(
                getattr(batch, "req_pool_indices_cpu", None),
                getattr(batch, "req_pool_indices", None),
            )
        ),
        "prefix_lens": _to_int_list(getattr(batch, "prefix_lens", None)),
        "extend_lens": _to_int_list(getattr(batch, "extend_lens", None)),
        "extend_num_tokens": _json_value(getattr(batch, "extend_num_tokens", None)),
        "seq_lens_sum": _json_value(getattr(batch, "seq_lens_sum", None)),
        "out_cache_loc_len": _safe_len(getattr(batch, "out_cache_loc", None)),
        "requests": [_summarize_req(r) for r in reqs[:MAX_SUMMARY_ITEMS]],
    }


def _summarize_waiting_queue(waiting_queue: List["Req"]) -> dict:
    return {
        "size": len(waiting_queue),
        "rids": [r.rid for r in waiting_queue[:MAX_SUMMARY_ITEMS]],
        "requests": [_summarize_req(r) for r in waiting_queue[:MAX_SUMMARY_ITEMS]],
    }


def _summarize_req(req: "Req") -> dict:
    sampling_params = getattr(req, "sampling_params", None)
    return {
        "rid": getattr(req, "rid", None),
        "req_pool_idx": _json_value(getattr(req, "req_pool_idx", None)),
        "input_len": _safe_len(getattr(req, "origin_input_ids", None)),
        "output_len": _safe_len(getattr(req, "output_ids", None)),
        "prefix_len": _safe_len(getattr(req, "prefix_indices", None)),
        "extend_input_len": _json_value(getattr(req, "extend_input_len", None)),
        "priority": _json_value(getattr(req, "priority", None)),
        "finished": getattr(req, "finished_reason", None) is not None,
        "max_new_tokens": _json_value(getattr(sampling_params, "max_new_tokens", None)),
    }


def _summarize_req_to_token_pool(
    pool: Optional[Any], running_batch: Optional["ScheduleBatch"] = None
) -> Optional[dict]:
    if pool is None:
        return None

    size = _json_value(getattr(pool, "size", None))
    available = _safe_call(pool, "available_size")
    free_slots = getattr(pool, "free_slots", None)
    data = {
        "class": type(pool).__name__,
        "size": size,
        "alloc_size": _json_value(getattr(pool, "_alloc_size", None)),
        "available_size": available,
        "used_size": _sub_ints(size, available),
        "max_context_len": _json_value(getattr(pool, "max_context_len", None)),
        "device": _json_value(getattr(pool, "device", None)),
        "free_slots_len": _safe_len(free_slots),
        "free_slots_head": _list_sample(free_slots, MAX_TOKEN_LOC_SAMPLE_ITEMS),
        "free_slots_tail": _list_tail_sample(free_slots, MAX_TOKEN_LOC_SAMPLE_ITEMS),
        "req_to_token_shape": _shape(getattr(pool, "req_to_token", None)),
        "active_rows": _summarize_req_to_token_active_rows(pool, running_batch),
    }

    mamba_pool = getattr(pool, "mamba_pool", None)
    mamba_allocator = getattr(pool, "mamba_allocator", None)
    if mamba_pool is not None or mamba_allocator is not None:
        mamba_size = _json_value(getattr(mamba_pool, "size", None))
        mamba_available = _safe_call(mamba_allocator, "available_size")
        data["mamba"] = {
            "size": mamba_size,
            "available_size": mamba_available,
            "used_size": _sub_ints(mamba_size, mamba_available),
        }
    return data


def _summarize_req_to_token_active_rows(
    pool: Any, running_batch: Optional["ScheduleBatch"]
) -> Optional[List[dict]]:
    if running_batch is None:
        return None

    table = getattr(pool, "req_to_token", None)
    if table is None:
        return None

    reqs = getattr(running_batch, "reqs", [])
    seq_lens = _to_int_list(
        _first_not_none(
            getattr(running_batch, "seq_lens_cpu", None),
            getattr(running_batch, "seq_lens", None),
        )
    )
    req_pool_indices = _to_int_list(
        _first_not_none(
            getattr(running_batch, "req_pool_indices_cpu", None),
            getattr(running_batch, "req_pool_indices", None),
        )
    )

    rows = []
    for i, req in enumerate(reqs[:MAX_SUMMARY_ITEMS]):
        req_pool_idx = getattr(req, "req_pool_idx", None)
        if (
            req_pool_idx is None
            and req_pool_indices is not None
            and i < len(req_pool_indices)
        ):
            req_pool_idx = req_pool_indices[i]
        req_pool_idx = _json_value(req_pool_idx)
        if not isinstance(req_pool_idx, int):
            rows.append(
                {
                    "rid": getattr(req, "rid", None),
                    "req_pool_idx": req_pool_idx,
                    "seq_len": _seq_len_for_req(req, seq_lens, i),
                    "token_locs_len": None,
                    "token_locs_head": None,
                    "token_locs_tail": None,
                }
            )
            continue

        seq_len = _seq_len_for_req(req, seq_lens, i)
        head, tail, token_locs_len = _sample_req_to_token_row(
            table, req_pool_idx, seq_len
        )
        rows.append(
            {
                "rid": getattr(req, "rid", None),
                "req_pool_idx": req_pool_idx,
                "seq_len": seq_len,
                "token_locs_len": token_locs_len,
                "token_locs_head": head,
                "token_locs_tail": tail,
            }
        )

    return rows


def _seq_len_for_req(req: "Req", seq_lens: Optional[List[int]], index: int) -> int:
    if seq_lens is not None and index < len(seq_lens):
        return seq_lens[index]
    return (_safe_len(getattr(req, "origin_input_ids", None)) or 0) + (
        _safe_len(getattr(req, "output_ids", None)) or 0
    )


def _sample_req_to_token_row(table: Any, row_idx: int, seq_len: int):
    if seq_len <= 0:
        return [], [], 0

    try:
        row_capacity = int(table.shape[1])
    except Exception:
        row_capacity = seq_len

    token_locs_len = min(seq_len, row_capacity)
    head_end = min(token_locs_len, MAX_TOKEN_LOC_SAMPLE_ITEMS)
    tail_start = max(0, token_locs_len - MAX_TOKEN_LOC_SAMPLE_ITEMS)

    try:
        head = _to_int_list(table[row_idx, :head_end])
        tail = _to_int_list(table[row_idx, tail_start:token_locs_len])
        return head, tail, token_locs_len
    except Exception:
        return None, None, token_locs_len


def _summarize_token_to_kv_pool_allocator(pool: Optional[Any]) -> Optional[dict]:
    if pool is None:
        return None

    size = _json_value(getattr(pool, "size", None))
    available = _safe_call(pool, "available_size")
    return {
        "class": type(pool).__name__,
        "size": size,
        "available_size": available,
        "used_size": _sub_ints(size, available),
        "full_available_size": _safe_call(pool, "full_available_size"),
        "swa_available_size": _safe_call(pool, "swa_available_size"),
        "page_size": _json_value(getattr(pool, "page_size", None)),
    }


def _summarize_prefix_cache(cache: Optional[Any]) -> Optional[dict]:
    if cache is None:
        return None

    root_node = getattr(cache, "root_node", None)
    children = getattr(root_node, "children", None)
    return {
        "class": type(cache).__name__,
        "disable": _json_value(getattr(cache, "disable", None)),
        "total_size": _safe_call(cache, "total_size"),
        "evictable_size": _safe_call(cache, "evictable_size"),
        "protected_size": _safe_call(cache, "protected_size"),
        "full_evictable_size": _safe_call(cache, "full_evictable_size"),
        "swa_evictable_size": _safe_call(cache, "swa_evictable_size"),
        "page_size": _json_value(getattr(cache, "page_size", None)),
        "root_children": _safe_len(children),
        "is_tree_cache": _safe_call(cache, "is_tree_cache"),
        "is_chunk_cache": _safe_call(cache, "is_chunk_cache"),
        "supports_mamba": _safe_call(cache, "supports_mamba"),
    }


def _first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _enum_name(value: Any) -> Optional[str]:
    if value is None:
        return None
    return getattr(value, "name", str(value))


def _safe_call(obj: Any, method_name: str) -> Any:
    if obj is None:
        return None
    method = getattr(obj, method_name, None)
    if method is None:
        return None
    try:
        return _json_value(method())
    except Exception:
        return None


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def _sub_ints(lhs: Any, rhs: Any) -> Optional[int]:
    if isinstance(lhs, int) and isinstance(rhs, int):
        return lhs - rhs
    return None


def _safe_len(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return len(value)
    except Exception:
        return None


def _shape(value: Any) -> Optional[List[int]]:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return [int(x) for x in shape]
    except Exception:
        return None


def _to_int_list(value: Any) -> Optional[List[int]]:
    if value is None:
        return None
    try:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "tolist"):
            value = value.tolist()
        return [int(x) for x in list(value)[:MAX_VECTOR_ITEMS]]
    except Exception:
        return None


def _list_sample(value: Any, limit: int) -> Optional[List[int]]:
    if value is None:
        return None
    try:
        return [int(x) for x in list(value)[:limit]]
    except Exception:
        return None


def _list_tail_sample(value: Any, limit: int) -> Optional[List[int]]:
    if value is None:
        return None
    try:
        values = list(value)
        return [int(x) for x in values[-limit:]]
    except Exception:
        return None
