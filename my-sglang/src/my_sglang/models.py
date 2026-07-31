"""调度器各层共享的数据模型。

阅读顺序：``Req`` 保存跨轮请求状态，``BatchForward`` 是一次模型调用的
只读快照，``MemorySnapshot`` 用于观察当前内存账本。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class RequestStatus(str, Enum):
    """请求生命周期状态。"""

    # 尚未获得请求行和本轮 KV 资源。
    WAITING = "waiting"
    # Prompt 只完成了部分 chunk，下一轮还需继续 EXTEND。
    PREFILLING = "prefilling"
    # Prompt 已完成，可以逐 token DECODE。
    RUNNING = "running"
    # 正常结束或 abort；不再参与调度。
    FINISHED = "finished"


class ForwardMode(str, Enum):
    """一次模型 forward 的两种教学模式。"""

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

    req_pool_idx: int  # token 属于哪个请求行。
    producer_job_id: int  # 哪个在途 job 会产生 token。
    output_index: int  # token 位于该 job 输出 batch 的第几行。


@dataclass(frozen=True)
class SamplingParams:
    """本项目实际使用的最小采样配置。"""

    max_new_tokens: int  # 最多允许生成多少个新 token。
    eos_token_ids: frozenset[int] = field(default_factory=frozenset)  # EOS 集合。

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")


@dataclass(eq=False)
class Req:
    """一个请求在排队、prefill 和 decode 期间共享的可变状态。"""

    # 请求唯一标识。
    rid: str
    # 用户输入的原始 prompt token。
    origin_input_ids: list[int]
    # 最大生成长度、EOS 等采样配置。
    sampling_params: SamplingParams
    # 模型已经生成的输出 token。
    output_ids: list[int] = field(default_factory=list)
    # 请求当前所处的调度阶段。
    status: RequestStatus = RequestStatus.WAITING
    # 请求在 ReqToTokenPool 中占用的行号；未绑定时为 None。
    req_pool_idx: int | None = None

    # 从 radix cache 命中、当前请求正在复用的 KV slot。
    prefix_indices: np.ndarray = field(
        default_factory=lambda: np.empty((0,), dtype=np.int64)
    )
    # radix cache 命中前缀的最后一个节点，用于后续解锁。
    last_node: Any | None = None
    # 当前由 radix cache 保护、不能淘汰的前缀 token 数。
    cache_protected_len: int = 0

    # 本轮 prefill 计划填到的绝对 token 长度。
    fill_len: int = 0
    # 本轮实际需要 extend 的 token 数。
    extend_input_len: int = 0

    # 已经分配物理 KV slot 的 token 长度，可能暂时领先 committed。
    kv_allocated_len: int = 0
    # 已经成功完成模型计算并提交的 KV token 长度。
    kv_committed_len: int = 0
    # 是否曾因 KV 压力被 retract；用于下次准入时采用保守预算。
    retracted_stain: bool = False

    # 结束类型，例如 eos、length 或 abort；未结束时为 None。
    finish_reason: str | None = None
    # 结束时的补充说明，主要记录 abort 原因。
    finish_message: str | None = None

    def __post_init__(self) -> None:
        if not self.rid:
            raise ValueError("rid must be non-empty")
        if not self.origin_input_ids:
            raise ValueError("origin_input_ids must be non-empty")

    @property
    def generated_count(self) -> int:
        """当前已经生成的 token 数。"""
        return len(self.output_ids)

    @property
    def full_token_ids(self) -> list[int]:
        """原始 prompt 与已生成输出拼接后的完整 token 序列。"""
        return [*self.origin_input_ids, *self.output_ids]

    @property
    def fill_ids(self) -> list[int]:
        """需要 prefill 的 token；retract 后包含此前已经生成的输出。"""
        return self.full_token_ids

    @property
    def last_token_id(self) -> int:
        """完整 token 序列中的最后一个 token。"""
        return self.full_token_ids[-1]

    @property
    def remaining_new_tokens(self) -> int:
        """距离最大生成长度还可以生成的 token 数。"""
        return max(self.sampling_params.max_new_tokens - self.generated_count, 0)

    def append_output(self, token_id: int) -> None:
        """追加一个模型输出；已结束请求禁止继续写入。"""
        if self.status is RequestStatus.FINISHED:
            raise RuntimeError(f"cannot append output to finished request {self.rid}")
        self.output_ids.append(int(token_id))

    def mark_running(self) -> None:
        """将未结束请求切换到可 decode 状态。"""
        if self.status is not RequestStatus.FINISHED:
            self.status = RequestStatus.RUNNING

    def maybe_finish(self) -> bool:
        """检查 EOS 和长度上限，并返回请求是否已经结束。"""
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
        """以 abort 原因结束请求。"""
        self.status = RequestStatus.FINISHED
        self.finish_reason = "abort"
        self.finish_message = message

    def reset_for_retract(self) -> None:
        """保留逻辑输出，清空物理 KV 状态后回到等待队列。"""
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
    """``MiniScheduleBatch`` 生成的 runner 只读参数快照。

    所有 ``*_by_req`` 字段都与 ``reqs`` 按下标一一对应。
    """

    mode: ForwardMode  # EXTEND 或 DECODE。
    reqs: tuple[Req, ...]  # 本轮请求顺序。
    input_ids_by_req: tuple[tuple[int, ...], ...]  # 本轮送入模型的 token。
    req_pool_indices: tuple[int, ...]  # 请求对应的映射表行号。
    out_cache_locs: tuple[tuple[int, ...], ...]  # 本轮写入的物理 KV slot。
    seq_lens: tuple[int, ...]  # 本轮结束后的总序列长度。
    prefix_slot_ids_by_req: tuple[tuple[int, ...], ...] = ()  # 复用的前缀 slot。
    extend_lens: tuple[int, ...] = ()  # 本轮每个请求新增的 token 数。
    chunk_starts_by_req: tuple[int, ...] = ()  # 当前 chunk 的绝对起点。
    is_last_prefill_chunk_by_req: tuple[bool, ...] = ()  # 是否填完 prompt。
    input_future_refs_by_req: tuple[FutureTokenRef | None, ...] = ()  # 设备侧输入。

    @property
    def batch_size(self) -> int:
        """返回请求数量。"""
        return len(self.reqs)


@dataclass(frozen=True)
class MemorySnapshot:
    """一个调度时刻的内存统计，只用于观察，不参与资源所有权。"""

    free_tokens: int  # allocator 尚可分配的 slot 数。
    allocated_tokens: int  # allocator 已占用的 slot 数（按整页统计）。
    mapped_tokens: int  # ReqToTokenPool 中已写入映射的位置数。
    cache_evictable_tokens: int  # radix cache 中可淘汰的 slot 数。
    cache_protected_tokens: int  # 被活跃请求锁住的 cache slot 数。
    decode_reserved_tokens: int = 0  # admission 为未来 decode 预留的 slot 数。
