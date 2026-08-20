"""Lesson 4: RMSNorm combines a row reduction and elementwise fusion."""

from __future__ import annotations

import argparse

import numpy as np

from cpu_sim import (
    masked_load,
    masked_store,
    programs_1d,
    require_row_program_shape,
    trace_program,
)


def rmsnorm(
    values: np.ndarray, weight: np.ndarray, block_size: int, eps: float = 1e-5, trace: bool = False
) -> np.ndarray:
    require_row_program_shape(values, block_size)
    if weight.shape != (values.shape[1],):
        raise ValueError("weight must have shape (cols,)")
    if eps <= 0:
        raise ValueError("eps must be > 0")
    output = np.empty_like(values)
    # 所有行复用同一组列 offsets；不同 row 是不同的逻辑 program。
    program = next(programs_1d(values.shape[1], block_size))
    weight_block = masked_load(weight, program.offsets, program.mask)
    for row in range(values.shape[0]):
        if trace:
            print(trace_program(program, label=f"row={row}"))
        x = masked_load(values[row], program.offsets, program.mask)
        # 先在行内归约平方和，再把结果融合进逐元素缩放。
        inv_rms = 1.0 / np.sqrt(np.sum(x * x) / values.shape[1] + eps)
        masked_store(output[row], program.offsets, x * inv_rms * weight_block, program.mask)
    return output


def reference_rmsnorm(values: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    inv_rms = 1.0 / np.sqrt(np.mean(values * values, axis=1, keepdims=True) + eps)
    return values * inv_rms * weight


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--cols", type=int, default=257)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()

    if args.rows < 0:
        raise ValueError("rows must be >= 0")
    rng = np.random.default_rng(1)
    values = rng.normal(size=(args.rows, args.cols)).astype(np.float32)
    weight = rng.normal(size=args.cols).astype(np.float32)
    actual = rmsnorm(values, weight, args.block_size, args.eps, args.trace)
    np.testing.assert_allclose(actual, reference_rmsnorm(values, weight, args.eps), rtol=1e-5, atol=1e-6)
    print(f"PASS rmsnorm rows={args.rows} cols={args.cols} block_size={args.block_size}")


if __name__ == "__main__":
    main()
