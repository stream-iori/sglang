"""Scheduler 与模型执行后端之间的接口。

``RunnerProtocol`` 是同步路径；``LazyRunnerProtocol`` 把一次调用拆成
``start -> kick -> finalize``，让 overlap scheduler 能观察在途计算。
``SglangMlxRunnerAdapter`` 只负责协议转换，不参与调度决策。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Protocol


class RunnerProtocol(Protocol):
    """同步 runner 的结构化接口，fake runner 和 MLX adapter 都可实现。"""

    def prefill(
        self,
        req_id: str,
        new_token_ids: list[int],
        full_token_ids: list[int],
        prefix_slot_ids: list[int],
        new_slot_ids: list[int],
        req_pool_idx: int,
    ) -> int: ...

    def decode_batch(self, req_ids: list[str]) -> list[int]: ...

    def extend(
        self,
        req_id: str,
        new_token_ids: list[int],
        new_slot_ids: list[int],
    ) -> int: ...

    def remove_request(self, req_id: str) -> None: ...


class LazyRunnerProtocol(RunnerProtocol, Protocol):
    """Overlap runner 接口。

    ``start`` 构建计算并返回 handle，``kick`` 提交异步执行，``finalize``
    才把 token 带回 CPU。chained decode 直接依赖前一个 handle。
    """

    def prefill_start(
        self,
        req_id: str,
        new_token_ids: list[int],
        full_token_ids: list[int],
        prefix_slot_ids: list[int],
        new_slot_ids: list[int],
        req_pool_idx: int,
    ) -> Any: ...

    def prefill_kick(self, pending: Any) -> None: ...

    def prefill_finalize(self, pending: Any) -> int: ...

    def extend_start(
        self,
        req_id: str,
        new_token_ids: list[int],
        new_slot_ids: list[int],
    ) -> Any: ...

    def extend_kick(self, pending: Any) -> None: ...

    def extend_finalize(self, pending: Any) -> int: ...

    def decode_batch_start(self, req_ids: list[str]) -> Any: ...

    def decode_batch_start_chained(self, previous: Any) -> Any: ...

    def decode_batch_kick(self, pending: Any) -> None: ...

    def decode_batch_finalize(self, pending: Any) -> list[int]: ...

    def discard_pending(self, pending: Any) -> None: ...


def _ensure_sglang_source_importable() -> None:
    """让独立子项目可以直接导入同仓库的 ``python/sglang`` 源码。"""
    repo_python = Path(__file__).resolve().parents[3] / "python"
    if repo_python.exists():
        path = str(repo_python)
        if path not in sys.path:
            sys.path.insert(0, path)


class SglangMlxRunnerAdapter:
    """把 mini runner 协议转发到生产 ``MlxModelRunner``。

    Req 生命周期、admission 和 KV slot 所有权仍由 my-sglang 管理。
    """
    def __init__(
        self,
        model_path: str,
        *,  # 后面的参数必须使用关键字传参，避免调用方把配置值按位置传错。
        mem_fraction_static: float = 0.2,
        disable_radix_cache: bool = True,
        trust_remote_code: bool = True,
    ):
        _ensure_sglang_source_importable()
        # 延迟导入 SGLang：只有真实 MLX 路径才需要加载这些较重的依赖。
        from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner

        self._runner = MlxModelRunner(
            model_path=model_path,
            trust_remote_code=trust_remote_code,
            disable_radix_cache=disable_radix_cache,
            mem_fraction_static=mem_fraction_static,
        )
        # disable_radix_cache=True 时，MlxModelRunner 仍然需要初始化内部 cache 结构。
        self._runner.init_cache_pools(req_to_token_pool=None)

    def prefill(
        self,
        req_id: str,
        new_token_ids: list[int],
        full_token_ids: list[int],
        prefix_slot_ids: list[int],
        new_slot_ids: list[int],
        req_pool_idx: int,
    ) -> int:
        # adapter 边界见 my-sglang/docs/dynamic-flows.md#runner-boundary。
        return self._runner.prefill(
            req_id=req_id,
            new_token_ids=new_token_ids,
            full_token_ids=full_token_ids,
            prefix_slot_ids=prefix_slot_ids,
            new_slot_ids=new_slot_ids,
            req_pool_idx=req_pool_idx,
        )

    def decode_batch(self, req_ids: list[str]) -> list[int]:
        # batch 的输入输出契约见 my-sglang/docs/data-structures.md#batch-forward。
        return self._runner.decode_batch(req_ids)

    def extend(
        self,
        req_id: str,
        new_token_ids: list[int],
        new_slot_ids: list[int],
    ) -> int:
        # prefill/extend 分流见 my-sglang/docs/dynamic-flows.md#chunked-flow。
        return self._runner.extend(
            req_id=req_id,
            new_token_ids=new_token_ids,
            new_slot_ids=new_slot_ids,
        )

    def prefill_start(
        self,
        req_id: str,
        new_token_ids: list[int],
        full_token_ids: list[int],
        prefix_slot_ids: list[int],
        new_slot_ids: list[int],
        req_pool_idx: int,
    ) -> Any:
        # lazy runner 时间线见 my-sglang/docs/dynamic-flows.md#overlap-flow。
        return self._runner.prefill_start(
            req_id=req_id,
            new_token_ids=new_token_ids,
            full_token_ids=full_token_ids,
            prefix_slot_ids=prefix_slot_ids,
            new_slot_ids=new_slot_ids,
            req_pool_idx=req_pool_idx,
        )

    def prefill_kick(self, pending: Any) -> None:
        # MLX 是 lazy execution；async_eval 会把 lazy token 交给后端排队执行。
        import mlx.core as mx

        mx.async_eval(pending.lazy_token)

    def prefill_finalize(self, pending: Any) -> int:
        return self._runner.prefill_finalize(pending)

    def extend_start(
        self,
        req_id: str,
        new_token_ids: list[int],
        new_slot_ids: list[int],
    ) -> Any:
        return self._runner.extend_start(
            req_id=req_id,
            new_token_ids=new_token_ids,
            new_slot_ids=new_slot_ids,
        )

    def extend_kick(self, pending: Any) -> None:
        import mlx.core as mx

        mx.async_eval(pending.lazy_token)

    def extend_finalize(self, pending: Any) -> int:
        return self._runner.extend_finalize(pending)

    def decode_batch_start(self, req_ids: list[str]) -> Any:
        return self._runner.decode_batch_start(req_ids)

    def decode_batch_start_chained(self, previous: Any) -> Any:
        # 直接复用生产 MlxModelRunner 的依赖链：下一轮 lazy graph 读取上一轮
        # lazy_tokens，不要求 scheduler 先把上一 token 同步回 Python。
        return self._runner.decode_batch_start_chained(previous)

    def decode_batch_kick(self, pending: Any) -> None:
        # decode_batch_start 返回的 lazy_tokens 是这一批请求的下一 token。
        import mlx.core as mx

        mx.async_eval(pending.lazy_tokens)

    def decode_batch_finalize(self, pending: Any) -> list[int]:
        return self._runner.decode_batch_finalize(pending)

    def discard_pending(self, pending: Any) -> None:
        # 已交给 MLX 的 lazy graph 不能可靠取消；只同步它，不把 token 提交到
        # runner 的逻辑 token 列表。随后 scheduler 会 remove_request/reset。
        import mlx.core as mx

        if hasattr(pending, "lazy_tokens"):
            mx.eval(pending.lazy_tokens)
        elif hasattr(pending, "lazy_token"):
            mx.eval(pending.lazy_token)

    def remove_request(self, req_id: str) -> None:
        self._runner.remove_request(req_id)

    def has_request(self, req_id: str) -> bool:
        # 这个方法主要给集成测试用，用来确认请求结束后 MLX runner 没有残留状态。
        return self._runner.has_request(req_id)
