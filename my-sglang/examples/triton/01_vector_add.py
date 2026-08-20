"""Lesson 1: a CPU simulation of Triton's 1-D vector-add indexing."""

from __future__ import annotations

import argparse

import numpy as np

from cpu_sim import masked_load, masked_store, programs_1d, trace_program


def vector_add(x: np.ndarray, y: np.ndarray, block_size: int, trace: bool = False) -> np.ndarray:
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError("x and y must be 1-D arrays with the same shape")
    output = np.empty_like(x)
    # 每次循环代表一个 Triton program，不是一个 CPU/GPU thread。
    for program in programs_1d(x.size, block_size):
        if trace:
            print(trace_program(program))
        # 尾块的无效 offsets 通过 mask 读成 0，随后也不会被 store。
        x_block = masked_load(x, program.offsets, program.mask)
        y_block = masked_load(y, program.offsets, program.mask)
        masked_store(output, program.offsets, x_block + y_block, program.mask)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1003)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()

    if args.n < 0:
        raise ValueError("n must be >= 0")
    x = np.arange(args.n, dtype=np.float32)
    y = np.arange(args.n, 0, -1, dtype=np.float32)
    actual = vector_add(x, y, args.block_size, args.trace)
    expected = x + y
    np.testing.assert_allclose(actual, expected)
    print(f"PASS vector_add n={args.n} block_size={args.block_size}")


if __name__ == "__main__":
    main()
