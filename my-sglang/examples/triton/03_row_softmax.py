"""Lesson 3: row-wise softmax with strides and reductions."""

from __future__ import annotations

import argparse

import torch
import triton
import triton.language as tl

from runtime_utils import (
    runtime_label,
    synchronize,
    torch_device,
    validate_runtime_tensors,
)


@triton.jit
def row_softmax_kernel(
    input_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """One program loads, reduces, and normalizes one complete matrix row."""
    row_idx = tl.program_id(axis=0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    col_mask = col_offsets < n_cols

    # stride 是相邻两行起点之间相隔的元素数，不是字节数。
    input_row_ptr = input_ptr + row_idx * input_row_stride
    row = tl.load(input_row_ptr + col_offsets, mask=col_mask, other=-float("inf"))

    # 先减最大值避免 exp 溢出；tl.max/tl.sum 都在当前 program 的 tile 内归约。
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax = numerator / denominator

    output_row_ptr = output_ptr + row_idx * output_row_stride
    tl.store(output_row_ptr + col_offsets, softmax, mask=col_mask)


def row_softmax(values: torch.Tensor, num_warps: int) -> torch.Tensor:
    validate_runtime_tensors(values)
    if values.ndim != 2:
        raise ValueError("values must be a 2-D tensor")
    if values.shape[0] <= 0 or values.shape[1] <= 0:
        raise ValueError("values must have non-empty rows and columns")
    if not values.is_contiguous():
        raise ValueError("values must be contiguous")
    if num_warps <= 0:
        raise ValueError("num_warps must be positive")

    n_rows, n_cols = values.shape
    block_size = triton.next_power_of_2(n_cols)
    if block_size > 65_536:
        raise ValueError("this one-program-per-row lesson supports at most 65536 columns")

    output = torch.empty_like(values)
    # grid=(n_rows,)：每个 program_id 直接选择一行。
    row_softmax_kernel[(n_rows,)](
        values,
        output,
        values.stride(0),
        output.stride(0),
        n_cols,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1023)
    parser.add_argument("--cols", type=int, default=781)
    parser.add_argument("--num-warps", type=int, default=8)
    args = parser.parse_args()

    if args.rows <= 0 or args.cols <= 0:
        raise ValueError("--rows and --cols must be positive")
    if args.num_warps <= 0:
        raise ValueError("--num-warps must be positive")
    device = torch_device()

    torch.manual_seed(1)
    values = torch.randn(
        (args.rows, args.cols), device=device, dtype=torch.float32
    )
    actual = row_softmax(values, args.num_warps)
    expected = torch.softmax(values, dim=1)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    synchronize()
    print(
        "PASS row softmax "
        f"(runtime={runtime_label()}, rows={args.rows}, cols={args.cols}, "
        f"block_size={triton.next_power_of_2(args.cols)})"
    )


if __name__ == "__main__":
    main()
