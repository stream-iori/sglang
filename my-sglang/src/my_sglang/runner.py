from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Protocol


class RunnerProtocol(Protocol):
    # Protocol 是 Python 的“结构化接口”：只要对象有这些方法，就可以当 Runner 使用。
    # fake runner 和真实 MLX adapter 都实现这三个方法，调度器不关心底层模型细节。
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
    # LazyRunnerProtocol 是 overlap scheduler 需要的接口。
    # start 只构建/提交模型计算，不立刻取结果；finalize 才真正读取 token。
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

    def decode_batch_start(self, req_ids: list[str]) -> Any: ...

    def decode_batch_kick(self, pending: Any) -> None: ...

    def decode_batch_finalize(self, pending: Any) -> list[int]: ...


def _ensure_sglang_source_importable() -> None:
    # my-sglang 是独立子项目；这里把相邻的 ../python 加入 sys.path，
    # 这样可以直接复用当前仓库里的 sglang 源码，而不是依赖额外安装。
    repo_python = Path(__file__).resolve().parents[3] / "python"
    if repo_python.exists():
        path = str(repo_python)
        if path not in sys.path:
            sys.path.insert(0, path)


class SglangMlxRunnerAdapter:
    # 这个 adapter 只负责把 mini scheduler 的调用转发给 SGLang 的 MlxModelRunner。
    # 调度、Req 生命周期、KV slot 映射仍然在 my-sglang 自己的代码里完成。
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
        return self._runner.prefill(
            req_id=req_id,
            new_token_ids=new_token_ids,
            full_token_ids=full_token_ids,
            prefix_slot_ids=prefix_slot_ids,
            new_slot_ids=new_slot_ids,
            req_pool_idx=req_pool_idx,
        )

    def decode_batch(self, req_ids: list[str]) -> list[int]:
        return self._runner.decode_batch(req_ids)

    def extend(
        self,
        req_id: str,
        new_token_ids: list[int],
        new_slot_ids: list[int],
    ) -> int:
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

    def decode_batch_start(self, req_ids: list[str]) -> Any:
        return self._runner.decode_batch_start(req_ids)

    def decode_batch_kick(self, pending: Any) -> None:
        # decode_batch_start 返回的 lazy_tokens 是这一批请求的下一 token。
        import mlx.core as mx

        mx.async_eval(pending.lazy_tokens)

    def decode_batch_finalize(self, pending: Any) -> list[int]:
        return self._runner.decode_batch_finalize(pending)

    def remove_request(self, req_id: str) -> None:
        self._runner.remove_request(req_id)

    def has_request(self, req_id: str) -> bool:
        # 这个方法主要给集成测试用，用来确认请求结束后 MLX runner 没有残留状态。
        return self._runner.has_request(req_id)
