from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class RequestStatus(str, Enum):
    # 请求在 mini runtime 里的三种生命周期状态。
    # WAITING: 已进入调度器，但还没有做 prefill。
    # RUNNING: 已完成 prefill，后续每轮 decode 一个 token。
    # FINISHED: 已命中 eos 或 max_new_tokens，可以释放资源。
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass(frozen=True)
class SamplingParams:
    # frozen=True 表示这个配置对象创建后不再修改，避免生成过程中参数漂移。
    # 它只限制 output，不包含 prompt
    max_new_tokens: int
    # default_factory 用来为每个实例创建独立的空集合，避免多个请求共享同一个可变对象。
    eos_token_ids: frozenset[int] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        # dataclass 创建对象后会自动调用 __post_init__，适合放参数校验。
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")


@dataclass
class Req:
    # rid 是请求 ID。真实 SGLang 会用它贯穿 tokenizer、scheduler、detokenizer。
    rid: str
    # prompt 已经被 tokenizer 编成 token id；mini runtime 不在 Req 里保存原始文本。
    origin_input_ids: list[int]
    sampling_params: SamplingParams
    # output_ids 只保存模型新生成的 token，不包含 prompt。
    output_ids: list[int] = field(default_factory=list)
    status: RequestStatus = RequestStatus.WAITING
    # req_pool_idx 模拟 SGLang req_to_token_pool 中的请求行号。
    req_pool_idx: int | None = None
    # kv_slots 记录这个请求逻辑上使用的所有 KV cache slot，包含 cache 命中的 prefix。
    kv_slots: list[int] = field(default_factory=list)
    # prefix_slot_ids 是从 radix cache 借用的 slot；请求结束时不能释放。
    prefix_slot_ids: list[int] = field(default_factory=list)
    # owned_kv_slots 是本请求新分配的 slot；结束时要么释放，要么交给 radix cache 接管。
    owned_kv_slots: list[int] = field(default_factory=list)
    finish_reason: str | None = None

    def __post_init__(self) -> None:
        # 这里尽早拒绝非法请求，避免调度器进入一半才发现状态不完整。
        if not self.rid:
            raise ValueError("rid must be non-empty")
        if not self.origin_input_ids:
            raise ValueError("origin_input_ids must be non-empty")

    @property
    def generated_count(self) -> int:
        return len(self.output_ids)

    @property
    def full_token_ids(self) -> list[int]:
        # 模型上下文 = prompt token + 已生成 token。
        return [*self.origin_input_ids, *self.output_ids]

    @property
    def last_token_id(self) -> int:
        # decode 阶段每轮只喂上一次生成的最后一个 token。
        return self.full_token_ids[-1]

    def append_output(self, token_id: int) -> None:
        # finished 请求不允许再追加 token，这是请求生命周期的基本保护。
        if self.status is RequestStatus.FINISHED:
            raise RuntimeError(f"cannot append output to finished request {self.rid}")
        self.output_ids.append(int(token_id))

    def mark_running(self) -> None:
        if self.status is not RequestStatus.FINISHED:
            self.status = RequestStatus.RUNNING

    def maybe_finish(self) -> bool:
        # 每次生成一个 token 后都检查停止条件；真实 SGLang 也会在 batch 结果处理阶段做类似判断。
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


@dataclass(frozen=True)
class BatchForward:
    # BatchForward 是一次模型调用前整理好的批数据，类似 SGLang 的 ForwardBatch 简化版。
    # frozen=True 表示 batch 创建后不再被调度器改写，便于测试和 trace。
    mode: str
    # reqs 保存这次 forward 覆盖的请求对象，顺序必须和下面的输入字段一致。
    reqs: tuple[Req, ...]
    # prefill 时是未命中 radix cache 的 suffix；decode 时每个请求只有一个 token。
    input_ids_by_req: tuple[tuple[int, ...], ...]
    # 每个请求在 ReqPool 里的行号。
    req_pool_indices: tuple[int, ...]
    # 本轮输入 token 写入 KV cache 的 slot。
    out_cache_locs: tuple[tuple[int, ...], ...]
    # 每个请求当前完整序列长度，用来观察 prefill/decode 状态。
    seq_lens: tuple[int, ...]
    # prefill 命中的 prefix slots；decode batch 不使用这个字段。
    prefix_slot_ids_by_req: tuple[tuple[int, ...], ...] = ()

    @property
    def batch_size(self) -> int:
        return len(self.reqs)
