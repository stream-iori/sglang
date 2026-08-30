"""CPU Fake CUDA runner：保留 CUDA overlap 的可观察依赖，不模拟性能。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Protocol, TYPE_CHECKING

import numpy as np

from my_sglang.models import ForwardBatch, ForwardMode

if TYPE_CHECKING:
    from my_sglang.overlap_scheduler import FutureMap
    from my_sglang.pools import ReqToTokenPool


class RunnerProtocol(Protocol):
    """同步 scheduler 与模型执行层之间的最小协议。"""

    def run_batch(
        self, forward: ForwardBatch, req_to_token_pool: "ReqToTokenPool"
    ) -> list[int]: ...
    def remove_request(self, req_id: str) -> None: ...


class FakeCudaStream:
    """CPU 上的 FIFO CUDA stream；任务只在 event synchronize 时推进。"""

    def __init__(self, name: str, trace: list[str]):
        self.name, self.trace = name, trace
        self._tasks: deque[tuple[str, Callable[[], None]]] = deque()

    def enqueue(self, label: str, task: Callable[[], None]) -> None:
        self._tasks.append((label, task))
        self.trace.append(f"{self.name}:enqueue:{label}")

    def run_until(self, marker: "FakeCudaEvent") -> None:
        while not marker.ready:
            if not self._tasks:
                raise RuntimeError(f"{self.name} cannot satisfy event {marker.label}")
            label, task = self._tasks.popleft()
            self.trace.append(f"{self.name}:run:{label}")
            task()

    def run_all(self) -> None:
        while self._tasks:
            label, task = self._tasks.popleft()
            self.trace.append(f"{self.name}:run:{label}")
            task()


class FakeCudaEvent:
    """record 后由所属 stream 的任务置 ready；synchronize 只推进依赖链。"""

    def __init__(self, label: str, trace: list[str]):
        self.label, self.trace, self.ready = label, trace, False
        self._producer: FakeCudaStream | None = None

    def record(self, stream: FakeCudaStream) -> None:
        self._producer = stream
        stream.enqueue(f"record:{self.label}", self._mark_ready)

    def _mark_ready(self) -> None:
        self.ready = True
        self.trace.append(f"event:ready:{self.label}")

    def synchronize(self) -> None:
        self.trace.append(f"event:sync:{self.label}")
        if not self.ready:
            if self._producer is None:
                raise RuntimeError(f"event {self.label} was never recorded")
            self._producer.run_until(self)


@dataclass
class FakeGenerationBatchResult:
    """CUDA ``GenerationBatchResult`` 的教学子集。"""

    label: str
    next_token_ids: np.ndarray
    copy_done: FakeCudaEvent
    _discarded: bool = False

    def resolve_cpu_tokens(self) -> list[int]:
        self.copy_done.synchronize()
        return [int(token) for token in self.next_token_ids]

    def discard(self) -> None:
        self.copy_done.synchronize()
        self._discarded = True


@dataclass
class FakeCudaRunner:
    """确定性 token 脚本 + CUDA-shaped forward/copy stream。

    ``run_batch_async`` 不运行任务：它只入队。CPU 读取旧 result 时才沿 event
    依赖推进，因而能断言 launch B1 早于 resolve B0。
    """

    tokens: list[int] = field(default_factory=list)
    fail_resolve_at: int | None = None
    trace: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.tokens = list(self.tokens)
        if not self.tokens:
            self.tokens = [100, 101, 102, 103]
        self.forward_stream = FakeCudaStream("forward", self.trace)
        self.copy_stream = FakeCudaStream("copy", self.trace)
        self._result_ct = 0
        self._resolve_ct = 0

    def run_batch_async(self, forward: ForwardBatch, future_map: "FutureMap") -> FakeGenerationBatchResult:
        result_id = self._result_ct
        self._result_ct += 1
        label = f"B{result_id}"
        device_tokens = np.empty((forward.batch_size,), dtype=np.int64)
        forward_done = FakeCudaEvent(f"{label}.forward_done", self.trace)
        copy_done = FakeCudaEvent(f"{label}.copy_done", self.trace)
        result = FakeGenerationBatchResult(label, device_tokens, copy_done)

        def forward_and_sample() -> None:
            if forward.forward_mode is ForwardMode.DECODE:
                inputs = future_map.gather(np.asarray(forward.req_pool_indices, dtype=np.int64))
                self.trace.append(f"forward:gather:{label}:{inputs.tolist()}")
            else:
                self.trace.append(f"forward:prefill:{label}")
            if len(self.tokens) < forward.batch_size:
                raise RuntimeError("fake token script exhausted")
            sampled = np.asarray(self.tokens[: forward.batch_size], dtype=np.int64)
            del self.tokens[: forward.batch_size]
            device_tokens[:] = sampled
            future_map.stash(np.asarray(forward.req_pool_indices, dtype=np.int64), device_tokens)
            self.trace.append(f"forward:sample:{label}:{sampled.tolist()}")

        self.forward_stream.enqueue(f"forward+sample:{label}", forward_and_sample)
        forward_done.record(self.forward_stream)

        def d2h() -> None:
            forward_done.synchronize()
            result.next_token_ids = device_tokens.copy()
            self.trace.append(f"copy:d2h:{label}:{result.next_token_ids.tolist()}")

        self.copy_stream.enqueue(f"d2h:{label}", d2h)
        copy_done.record(self.copy_stream)
        return result

    def resolve(self, result: FakeGenerationBatchResult) -> list[int]:
        self._resolve_ct += 1
        if self.fail_resolve_at == self._resolve_ct:
            raise RuntimeError("injected fake CUDA resolve failure")
        return result.resolve_cpu_tokens()

    # 同步路径继续使用同一 token script，但接收完整 ForwardBatch。
    def _next(self, count: int = 1) -> list[int]:
        if len(self.tokens) < count:
            raise RuntimeError("fake token script exhausted")
        out, self.tokens = self.tokens[:count], self.tokens[count:]
        return out

    def run_batch(
        self, forward: ForwardBatch, req_to_token_pool: "ReqToTokenPool"
    ) -> list[int]:
        del req_to_token_pool
        sampled = self._next(forward.batch_size)
        self.trace.append(
            f"sync:sample:{forward.forward_mode.value}:{sampled}"
        )
        return sampled

    def remove_request(self, req_id: str) -> None:
        self.trace.append(f"runner:remove:{req_id}")
