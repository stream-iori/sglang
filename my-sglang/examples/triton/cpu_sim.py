"""CPU helpers that model Triton's indexing semantics, not GPU execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class ProgramSlice:
    """The logical elements owned by one 1-D Triton program."""

    pid: int
    offsets: np.ndarray
    mask: np.ndarray


def ceil_div(n: int, divisor: int) -> int:
    if n < 0:
        raise ValueError("n must be >= 0")
    if divisor <= 0:
        raise ValueError("block_size must be > 0")
    return (n + divisor - 1) // divisor


def programs_1d(n: int, block_size: int) -> Iterator[ProgramSlice]:
    """Yield the CPU equivalent of a 1-D Triton launch grid."""

    # grid 的 CPU 对应物：依次执行每一个逻辑 program。
    for pid in range(ceil_div(n, block_size)):
        # tl.arange 生成块内逻辑下标；尾块会包含越界下标。
        offsets = pid * block_size + np.arange(block_size, dtype=np.int64)
        yield ProgramSlice(pid=pid, offsets=offsets, mask=offsets < n)


def masked_load(
    values: np.ndarray, offsets: np.ndarray, mask: np.ndarray, other: float = 0.0
) -> np.ndarray:
    """Equivalent in meaning to tl.load(ptr + offsets, mask=mask, other=other)."""

    if offsets.shape != mask.shape:
        raise ValueError("offsets and mask must have the same shape")
    # 先填充 other，再只读取 mask=True 的位置，避免 NumPy 越界访问。
    loaded = np.full(offsets.shape, other, dtype=values.dtype)
    loaded[mask] = values[offsets[mask]]
    return loaded


def masked_store(
    destination: np.ndarray,
    offsets: np.ndarray,
    values: np.ndarray,
    mask: np.ndarray,
) -> None:
    """Equivalent in meaning to tl.store(ptr + offsets, values, mask=mask)."""

    if offsets.shape != values.shape or offsets.shape != mask.shape:
        raise ValueError("offsets, values, and mask must have the same shape")
    # mask=False 的 lane 不写回，对应 tl.store(..., mask=mask)。
    destination[offsets[mask]] = values[mask]


def require_row_program_shape(values: np.ndarray, block_size: int) -> None:
    if values.ndim != 2:
        raise ValueError("values must be a 2-D array")
    if values.shape[1] <= 0:
        raise ValueError("cols must be > 0")
    if block_size <= 0:
        raise ValueError("block_size must be > 0")
    if block_size < values.shape[1]:
        raise ValueError("this lesson uses one program per row: block_size must be >= cols")


def trace_program(program: ProgramSlice, *, label: str = "") -> str:
    valid = program.offsets[program.mask]
    prefix = f"{label} " if label else ""
    if valid.size == 0:
        valid_range = "empty"
    else:
        valid_range = f"{valid[0]}..{valid[-1]}"
    return (
        f"{prefix}pid={program.pid} offsets={program.offsets[0]}.."
        f"{program.offsets[-1]} valid={valid_range}"
    )
