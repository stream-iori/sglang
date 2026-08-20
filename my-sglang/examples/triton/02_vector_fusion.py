"""Lesson 2: fuse add and multiply in one logical Triton program."""

from __future__ import annotations

import argparse

import numpy as np

from cpu_sim import masked_load, masked_store, programs_1d, trace_program


def fused_vector(x: np.ndarray, y: np.ndarray, scale: float, block_size: int, trace: bool = False) -> np.ndarray:
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError("x and y must be 1-D arrays with the same shape")
    output = np.empty_like(x)
    # 一个 program 内连续完成 add 和 multiply，语义上没有中间输出数组。
    for program in programs_1d(x.size, block_size):
        if trace:
            print(trace_program(program))
        x_block = masked_load(x, program.offsets, program.mask)
        y_block = masked_load(y, program.offsets, program.mask)
        fused = (x_block + y_block) * np.float32(scale)  # 临时值只在当前逻辑块中存在。
        masked_store(output, program.offsets, fused, program.mask)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1003)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--scale", type=float, default=1.5)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()

    if args.n < 0:
        raise ValueError("n must be >= 0")
    x = np.linspace(-2.0, 2.0, args.n, dtype=np.float32)
    y = np.linspace(3.0, -3.0, args.n, dtype=np.float32)
    actual = fused_vector(x, y, args.scale, args.block_size, args.trace)
    np.testing.assert_allclose(actual, (x + y) * np.float32(args.scale))
    print(f"PASS vector_fusion n={args.n} block_size={args.block_size} scale={args.scale}")


if __name__ == "__main__":
    main()
