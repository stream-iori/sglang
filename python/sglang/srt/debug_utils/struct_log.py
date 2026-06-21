"""用于运行时结构日志的轻量工具函数。

这个模块刻意保持轻量，并且不依赖 SGLang 的重模块，否则会自动引入 CUDA 和 Triton。它会被
TokenizerManager、Scheduler、MLX worker 这些高频路径导入，所以在调试开关
关闭时，正常路径的额外开销必须尽量低。

开启方式：

    SGLANG_DEBUG_STRUCT=1 python -m sglang.launch_server ...

日志内容是“摘要”，不是完整对象。SGLang 的运行时对象经常持有大 tensor、
token 数组、KV cache、请求文本等。直接打印完整对象会很慢、很吵，甚至会改变
多进程服务的时序。因此这里主要打印 type、rid、shape、dtype、前几个 token id
和关键长度。
"""

import logging
import os
from collections.abc import Iterable
from typing import Any


# 用作功能开关的环境变量名。集中成常量，避免每个调用点重复手写字符串。
DEBUG_ENV_VAR = "SGLANG_DEBUG_STRUCT"


def debug_struct_enabled() -> bool:
    """判断结构日志是否开启。

    os.getenv(...) 会读取当前进程的环境变量。SGLang 的子进程会继承启动
    server 时的环境变量，所以启动前设置 SGLANG_DEBUG_STRUCT=1，通常就能让
    TokenizerManager、Scheduler 等子进程一起输出结构日志。

    为了命令行使用方便，下面这些值都认为是开启：
    1、true、yes、on。
    """
    return os.getenv(DEBUG_ENV_VAR, "").lower() in ("1", "true", "yes", "on")


def shape_of(obj: Any) -> Any:
    """安全读取 obj.shape。

    getattr(obj, "shape", None) 的意思是：
    - 如果 obj 有名为 "shape" 的属性，就返回它；
    - 如果没有，就返回 None。

    这对调试 tensor、MLX array、numpy array 很有用，因为它们通常都有 shape；
    但普通 Python 对象不一定有。
    """
    return getattr(obj, "shape", None)


def dtype_of(obj: Any) -> Any:
    """安全读取 tensor 类对象的 obj.dtype。"""
    return getattr(obj, "dtype", None)


def short_list(values: Any, limit: int = 8) -> Any:
    """返回列表类对象的短预览。

    例子：
    - [1, 2, 3] 保持为 [1, 2, 3]
    - range(100) 会变成 [0, 1, ..., "...(+92)"]

    token id 或 request id 很长时，这样可以保持日志可读。字符串和 bytes 会原样
    返回，因为 list("hello") 会变成字符列表，对这里的调试没帮助。
    """
    if values is None:
        return None
    if isinstance(values, (str, bytes)):
        return values
    try:
        seq = list(values)
    except TypeError:
        return values
    if len(seq) <= limit:
        return seq
    return seq[:limit] + [f"...(+{len(seq) - limit})"]


def obj_keys(obj: Any, limit: int = 24) -> Any:
    """返回对象字段名的短列表。

    大多数普通 Python 对象会把实例属性放在 __dict__ 里。dataclass 类还会暴露
    __dataclass_fields__。这里仅打印字段名，不打印字段值，所以对大对象或包含
    敏感内容的对象更安全。
    """
    keys = None
    if hasattr(obj, "__dict__"):
        keys = list(obj.__dict__.keys())
    elif hasattr(obj, "__dataclass_fields__"):
        keys = list(obj.__dataclass_fields__.keys())
    if keys is None:
        return None
    return short_list(keys, limit)


def summarize_req(req: Any) -> dict[str, Any]:
    """摘要打印 Scheduler 里的 Req 对象。

    Req 是 Scheduler 内部的“单个用户请求”对象。学习运行时结构时，最常看的
    通常是：
    - rid：请求 id，用来跨进程串联日志；
    - input_len/output_len：prompt token 数和已生成 token 数；
    - prefix_len：命中的 prefix cache 长度；
    - req_pool_idx：该请求在 request-to-token pool 里的槽位。

    这里参数类型写 Any，而不是直接 import Req，是为了避免循环导入，也让这个
    调试工具不依赖 Scheduler 的具体实现。
    """
    origin_input_ids = getattr(req, "origin_input_ids", None)
    output_ids = getattr(req, "output_ids", None)
    prefix_indices = getattr(req, "prefix_indices", None)
    sampling_params = getattr(req, "sampling_params", None)
    return {
        "type": type(req).__name__,
        "rid": getattr(req, "rid", None),
        "input_len": len(origin_input_ids) if origin_input_ids is not None else None,
        "output_len": len(output_ids) if output_ids is not None else None,
        "prefix_len": len(prefix_indices) if prefix_indices is not None else None,
        "fill_len": getattr(req, "fill_len", None),
        "extend_input_len": getattr(req, "extend_input_len", None),
        "req_pool_idx": getattr(req, "req_pool_idx", None),
        "kv_committed_len": getattr(req, "kv_committed_len", None),
        "kv_allocated_len": getattr(req, "kv_allocated_len", None),
        "cache_protected_len": getattr(req, "cache_protected_len", None),
        "mamba_last_track_seqlen": getattr(req, "mamba_last_track_seqlen", None),
        "is_retracted": getattr(req, "is_retracted", None),
        "finished": (
            req.finished()
            if hasattr(req, "finished") and callable(getattr(req, "finished"))
            else None
        ),
        "finished_reason": str(getattr(req, "finished_reason", None)),
        "priority": getattr(req, "priority", None),
        "stream": getattr(req, "stream", None),
        "max_new_tokens": getattr(sampling_params, "max_new_tokens", None),
    }


def summarize_batch(batch: Any, req_limit: int = 4) -> dict[str, Any]:
    """摘要打印类似 ScheduleBatch 的对象。

    ScheduleBatch 是 SGLang 调度的核心结构。它把多个 Req 组织成一个 batch，
    并携带会送进 model worker 的 tensor 字段。这里仅打印前几个请求和 tensor
    元数据：
    - input_ids.shape/dtype，而不是全部 token id；
    - seq_lens 的值，这通常比较小而且很有用；
    - req_pool_indices，用来把请求关联到 memory pool 的行；
    - out_cache_loc.shape/dtype，用来观察 KV cache 分配形状。
    """
    reqs = getattr(batch, "reqs", None) or []
    forward_mode = getattr(batch, "forward_mode", None)
    return {
        "type": type(batch).__name__,
        "forward_iter": getattr(batch, "forward_iter", None),
        "forward_mode": str(forward_mode) if forward_mode is not None else None,
        "batch_size": len(reqs),
        "rids": short_list([getattr(req, "rid", None) for req in reqs], req_limit),
        "reqs": [summarize_req(req) for req in reqs[:req_limit]],
        "input_ids": {
            "shape": shape_of(getattr(batch, "input_ids", None)),
            "dtype": dtype_of(getattr(batch, "input_ids", None)),
        },
        "seq_lens": {
            "shape": shape_of(getattr(batch, "seq_lens", None)),
            "value": short_list(_safe_iter(getattr(batch, "seq_lens", None))),
        },
        "extend_lens": short_list(getattr(batch, "extend_lens", None)),
        "req_pool_indices": {
            "shape": shape_of(getattr(batch, "req_pool_indices", None)),
            "value": short_list(_safe_iter(getattr(batch, "req_pool_indices", None))),
        },
        "out_cache_loc": {
            "shape": shape_of(getattr(batch, "out_cache_loc", None)),
            "dtype": dtype_of(getattr(batch, "out_cache_loc", None)),
        },
    }


def summarize_tokenized_request(obj: Any) -> dict[str, Any]:
    """摘要打印 tokenization 之后、发送给 Scheduler 之前的请求。

    这是 HTTP/tokenizer 侧和 scheduler 侧之间的桥。input_ids_head 会展示前几个
    token id，让你不用打印完整 prompt 也能检查 tokenization 是否符合预期。
    """
    input_ids = getattr(obj, "input_ids", None)
    sampling_params = getattr(obj, "sampling_params", None)
    return {
        "type": type(obj).__name__,
        "rid": getattr(obj, "rid", None),
        "input_text_type": type(getattr(obj, "input_text", None)).__name__,
        "input_len": len(input_ids) if input_ids is not None else None,
        "input_ids_head": short_list(input_ids),
        "stream": getattr(obj, "stream", None),
        "return_logprob": getattr(obj, "return_logprob", None),
        "max_new_tokens": getattr(sampling_params, "max_new_tokens", None),
        "keys": obj_keys(obj),
    }


def log_struct(logger: logging.Logger, label: str, payload: Any) -> None:
    """当 SGLANG_DEBUG_STRUCT 开启时，打印一个已经构造好的 payload。

    适合用于调用点已经有的小字典等低成本内容。对于构造成本较高的摘要，优先
    使用下面的 log_struct_lazy。
    """
    if debug_struct_enabled():
        logger.info("[STRUCT] %s %s", label, payload)


def log_struct_lazy(logger: logging.Logger, label: str, payload_fn) -> None:
    """只在调试开关开启时才构造并打印 payload。

    payload_fn 是一个无参数函数，通常写成：

        lambda: summarize_batch(batch)

    这在高频路径里很重要。如果调试关闭，lambda 根本不会被调用，因此不会遍历
    请求、不会把 tensor 移到 CPU、也不会构造最后会被丢弃的字典。
    """
    if debug_struct_enabled():
        logger.info("[STRUCT] %s %s", label, payload_fn())


def _safe_iter(value: Any) -> Iterable[Any] | None:
    """尽力把 tensor 类对象转换成 Python list 类值。

    PyTorch tensor 可能在 GPU/MPS/CPU 上。日志里只需要小而可读的值，所以这里
    会尝试：
    - detach()：如果有 autograd 跟踪，先脱离计算图；
    - cpu()：如果支持，把 tensor 值移到 CPU；
    - tolist()：转换成普通 Python list。

    如果某一步不支持或失败，就返回 None，避免为了打日志影响 server 正常运行。
    """
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            return None
    return value
