from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class RequestStatus(str, Enum):
    WAITING = "waiting"
    PREFILLING = "prefilling"
    RUNNING = "running"
    FINISHED = "finished"


class ForwardMode(str, Enum):
    # 真实 SGLang 把 prompt/prefix 扩展统一称为 EXTEND，decode 单独成批。
    EXTEND = "extend"
    DECODE = "decode"


@dataclass(frozen=True)
class FutureTokenRef:
    """设备侧尚未回到 CPU、但可作为下一轮 decode 输入的 token 引用。

    它是教学用的依赖描述，不保存 token 值：``producer_job_id`` 指向生产它的
    batch，``output_index`` 指向该 batch 的输出行，``req_pool_idx`` 用于把行
    与请求的持久槽位对应起来。
    """

    req_pool_idx: int
    producer_job_id: int
    output_index: int


@dataclass(frozen=True)
class SamplingParams:
    max_new_tokens: int
    eos_token_ids: frozenset[int] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")


@dataclass(eq=False)
class Req:
    rid: str
    origin_input_ids: list[int]
    sampling_params: SamplingParams
    output_ids: list[int] = field(default_factory=list)
    status: RequestStatus = RequestStatus.WAITING
    req_pool_idx: int | None = None

    # prefix_indices 是从 radix cache 命中的、当前请求正在借用的 KV slot。
    prefix_indices: np.ndarray = field(
        default_factory=lambda: np.empty((0,), dtype=np.int64)
    )
    last_node: Any | None = None
    cache_protected_len: int = 0

    # fill_len/extend_input_len 描述本轮 prompt 或 re-prefill 上下文的范围。
    fill_len: int = 0
    extend_input_len: int = 0

    # allocated 可以领先 committed；overlap launch/finalize 之间会看到这个差异。
    kv_allocated_len: int = 0
    kv_committed_len: int = 0
    retracted_stain: bool = False

    finish_reason: str | None = None
    finish_message: str | None = None

    def __post_init__(self) -> None:
        if not self.rid:
            raise ValueError("rid must be non-empty")
        if not self.origin_input_ids:
            raise ValueError("origin_input_ids must be non-empty")

    @property
    def generated_count(self) -> int:
        return len(self.output_ids)

    @property
    def full_token_ids(self) -> list[int]:
        return [*self.origin_input_ids, *self.output_ids]

    @property
    def fill_ids(self) -> list[int]:
        # retract 后需要把已经生成的 token 一并重新 prefill。
        return self.full_token_ids

    @property
    def last_token_id(self) -> int:
        return self.full_token_ids[-1]

    @property
    def remaining_new_tokens(self) -> int:
        return max(self.sampling_params.max_new_tokens - self.generated_count, 0)

    def append_output(self, token_id: int) -> None:
        if self.status is RequestStatus.FINISHED:
            raise RuntimeError(f"cannot append output to finished request {self.rid}")
        self.output_ids.append(int(token_id))

    def mark_running(self) -> None:
        if self.status is not RequestStatus.FINISHED:
            self.status = RequestStatus.RUNNING

    def maybe_finish(self) -> bool:
        # 字段与状态变化图见 my-sglang/docs/data-structures.md#req-state。
        if self.status is RequestStatus.FINISHED:
            return True
        if (
            self.output_ids
            and self.output_ids[-1] in self.sampling_params.eos_token_ids
        ):
            self.status = RequestStatus.FINISHED
            self.finish_reason = "eos"
            return True
        if self.generated_count >= self.sampling_params.max_new_tokens:
            self.status = RequestStatus.FINISHED
            self.finish_reason = "length"
            return True
        return False

    def mark_aborted(self, message: str) -> None:
        self.status = RequestStatus.FINISHED
        self.finish_reason = "abort"
        self.finish_message = message

    def reset_for_retract(self) -> None:
        # 逻辑 token 保留；所有物理 KV/row/lock 状态必须清空后重新 admission。
        self.status = RequestStatus.WAITING
        self.req_pool_idx = None
        self.prefix_indices = np.empty((0,), dtype=np.int64)
        self.last_node = None
        self.cache_protected_len = 0
        self.fill_len = 0
        self.extend_input_len = 0
        self.kv_allocated_len = 0
        self.kv_committed_len = 0
        self.retracted_stain = True


@dataclass(frozen=True)
class BatchForward:
    # MiniScheduleBatch.to_forward_batch() 生成的、对 runner 友好的不可变快照。
    mode: ForwardMode
    reqs: tuple[Req, ...]
    input_ids_by_req: tuple[tuple[int, ...], ...]
    req_pool_indices: tuple[int, ...]
    out_cache_locs: tuple[tuple[int, ...], ...]
    seq_lens: tuple[int, ...]
    prefix_slot_ids_by_req: tuple[tuple[int, ...], ...] = ()
    extend_lens: tuple[int, ...] = ()
    chunk_starts_by_req: tuple[int, ...] = ()
    is_last_prefill_chunk_by_req: tuple[bool, ...] = ()
    input_future_refs_by_req: tuple[FutureTokenRef | None, ...] = ()

    @property
    def batch_size(self) -> int:
        return len(self.reqs)


@dataclass(frozen=True)
class MemorySnapshot:
    free_tokens: int
    allocated_tokens: int
    mapped_tokens: int
    cache_evictable_tokens: int
    cache_protected_tokens: int
    decode_reserved_tokens: int = 0
