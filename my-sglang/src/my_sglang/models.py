"""调度器各层共享的数据模型。

阅读顺序：``Req`` 保存跨轮请求状态，``ForwardBatch`` 是一次模型调用的
只读快照，``MemorySnapshot`` 用于观察当前内存账本。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, NamedTuple

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


class Range(NamedTuple):
    """与 SRT ``Range`` 相同的左闭右开请求内区间。"""

    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(slots=True, kw_only=True)
class ReqKvInfo:
    """与 SRT 相同层级的请求 KV 分配状态。"""

    kv_allocated_len: int = 0


class BaseFinishReason:
    """标准 SRT finish reason 的最小教学接口。"""

    def to_json(self) -> dict[str, object]:
        raise NotImplementedError


@dataclass(frozen=True)
class FINISH_MATCHED_TOKEN(BaseFinishReason):
    matched: int

    def to_json(self) -> dict[str, object]:
        return {"type": "stop", "matched": self.matched}


@dataclass(frozen=True)
class FINISH_LENGTH(BaseFinishReason):
    length: int

    def to_json(self) -> dict[str, object]:
        return {"type": "length", "length": self.length}


@dataclass(frozen=True)
class FINISH_ABORT(BaseFinishReason):
    message: str = "Aborted"

    def to_json(self) -> dict[str, object]:
        return {"type": "abort", "message": self.message}


@dataclass(frozen=True)
class SamplingParams:
    """本项目实际使用的最小采样配置。"""

    max_new_tokens: int  # 最多允许生成多少个新 token。
    stop_token_ids: frozenset[int] = field(default_factory=frozenset)

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
    # 模型 EOS 集合；用户指定 stop token 另存于 sampling_params。
    eos_token_ids: frozenset[int] = field(default_factory=frozenset)
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

    # 本轮 EXTEND 的请求内绝对区间。
    extend_range: Range | None = None

    # 已分配 KV 状态与标准 SRT 一样放在 ReqKvInfo 中。
    kv: ReqKvInfo = field(default_factory=ReqKvInfo)
    # 已经成功完成模型计算并提交的 KV token 长度。
    kv_committed_len: int = 0
    # 是否曾因 KV 压力被 retract；用于下次准入时采用保守预算。
    retracted_stain: bool = False

    # 标准 SRT 使用结构化 finish reason，而不是字符串状态。
    finished_reason: BaseFinishReason | None = None

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
    def full_untruncated_fill_ids(self) -> list[int]:
        """原始 prompt 与已生成输出拼接后的完整 token 序列。"""
        return [*self.origin_input_ids, *self.output_ids]

    def get_fill_ids(self) -> list[int]:
        """需要 prefill 的 token；retract 后包含此前已经生成的输出。"""
        return self.full_untruncated_fill_ids

    @property
    def last_token_id(self) -> int:
        """完整 token 序列中的最后一个 token。"""
        return self.get_fill_ids()[-1]

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
            and (
                self.output_ids[-1] in self.eos_token_ids
                or self.output_ids[-1] in self.sampling_params.stop_token_ids
            )
        ):
            self.status = RequestStatus.FINISHED
            self.finished_reason = FINISH_MATCHED_TOKEN(self.output_ids[-1])
            return True
        if self.generated_count >= self.sampling_params.max_new_tokens:
            self.status = RequestStatus.FINISHED
            self.finished_reason = FINISH_LENGTH(self.generated_count)
            return True
        return False

    def mark_aborted(self, message: str) -> None:
        """以 abort 原因结束请求。"""
        self.status = RequestStatus.FINISHED
        self.finished_reason = FINISH_ABORT(message)

    def reset_for_retract(self) -> None:
        """保留逻辑输出，清空物理 KV 状态后回到等待队列。"""
        # 逻辑 token 保留；所有物理 KV/row/lock 状态必须清空后重新 admission。
        self.status = RequestStatus.WAITING
        self.req_pool_idx = None
        self.prefix_indices = np.empty((0,), dtype=np.int64)
        self.last_node = None
        self.cache_protected_len = 0
        self.extend_range = None
        self.kv.kv_allocated_len = 0
        self.kv_committed_len = 0
        self.retracted_stain = True


@dataclass(frozen=True)
class ForwardBatch:
    """``MiniScheduleBatch`` 生成的 runner 只读参数快照。

    ``*_by_req`` 属性是从标准展平字段派生的教学观察视图。
    """

    forward_mode: ForwardMode  # EXTEND 或 DECODE。
    reqs: tuple[Req, ...]  # 本轮请求顺序。
    input_ids: tuple[int, ...]  # 与标准 SRT 一样展平的模型输入 token。
    req_pool_indices: tuple[int, ...]  # 请求对应的映射表行号。
    out_cache_loc: tuple[int, ...]  # 与 input_ids 对齐的展平物理 KV slot。
    seq_lens: tuple[int, ...]  # 本轮结束后的总序列长度。
    prefix_indices_by_req: tuple[tuple[int, ...], ...] = ()  # 复用的前缀 slot。
    extend_seq_lens: tuple[int, ...] = ()  # 本轮每个请求新增的 token 数。
    extend_range_starts: tuple[int, ...] = ()  # 请求内当前 EXTEND 起点。
    contains_last_prefill_chunk: bool = True

    @property
    def batch_size(self) -> int:
        """返回请求数量。"""
        return len(self.reqs)

    @property
    def seq_lens_sum(self) -> int:
        return sum(self.seq_lens)

    @property
    def input_ids_by_req(self) -> tuple[tuple[int, ...], ...]:
        """教学观察视图；标准 ForwardBatch 只保存展平 ``input_ids``。"""
        return self._split_flat(self.input_ids)

    @property
    def out_cache_loc_by_req(self) -> tuple[tuple[int, ...], ...]:
        """教学观察视图；标准 ForwardBatch 只保存展平 ``out_cache_loc``。"""
        return self._split_flat(self.out_cache_loc)

    def _split_flat(self, values: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
        groups: list[tuple[int, ...]] = []
        offset = 0
        for length in self.extend_seq_lens:
            groups.append(values[offset : offset + length])
            offset += length
        if offset != len(values):
            raise AssertionError(
                "flat ForwardBatch field does not match extend_seq_lens"
            )
        return tuple(groups)


@dataclass(frozen=True)
class MemorySnapshot:
    """一个调度时刻的内存统计，只用于观察，不参与资源所有权。"""

    free_tokens: int  # allocator 尚可分配的 slot 数。
    allocated_tokens: int  # allocator 已占用的 slot 数（按整页统计）。
    mapped_tokens: int  # ReqToTokenPool 中已写入映射的位置数。
    cache_evictable_tokens: int  # radix cache 中可淘汰的 slot 数。
    cache_protected_tokens: int  # 被活跃请求锁住的 cache slot 数。
    decode_reserved_tokens: int = 0  # admission 为未来 decode 预留的 slot 数。
