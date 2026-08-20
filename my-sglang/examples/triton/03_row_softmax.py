"""Lesson 3: one logical program per row with a masked reduction."""

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


def row_softmax(values: np.ndarray, block_size: int, trace: bool = False) -> np.ndarray:
    require_row_program_shape(values, block_size)
    output = np.empty_like(values)
    # 本课限制“一行一个 program”，所以 block_size 必须容纳整行。
    offsets = next(programs_1d(values.shape[1], block_size))
    for row in range(values.shape[0]):
        if trace:
            print(trace_program(offsets, label=f"row={row}"))
        # 无效 lane 填 -inf，使其不影响 max 和后续 exp/sum 归约。
        row_values = masked_load(values[row], offsets.offsets, offsets.mask, other=-np.inf)
        row_max = np.max(row_values)
        # 先减最大值，避免 exp(大数) 溢出。
        exp_values = np.exp(row_values - row_max)
        denominator = np.sum(exp_values)
        masked_store(output[row], offsets.offsets, exp_values / denominator, offsets.mask)
    return output


def reference_softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--cols", type=int, default=257)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()

    if args.rows < 0:
        raise ValueError("rows must be >= 0")
    rng = np.random.default_rng(0)
    values = rng.normal(size=(args.rows, args.cols)).astype(np.float32)
    if args.rows:
        values[0, 0] = 1000.0
        if args.cols > 1:
            values[0, 1] = -1000.0
    actual = row_softmax(values, args.block_size, args.trace)
    np.testing.assert_allclose(actual, reference_softmax(values), rtol=1e-6, atol=1e-6)
    print(f"PASS row_softmax rows={args.rows} cols={args.cols} block_size={args.block_size}")


if __name__ == "__main__":
    main()
