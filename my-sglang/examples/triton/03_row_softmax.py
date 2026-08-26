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
    # grid=(n_rows,)，所以 program_id 可直接作为当前要处理的矩阵行号。
    row_idx = tl.program_id(axis=0)
    # tl.arange 要求二次幂范围；BLOCK_SIZE 是不小于真实列数的最小二次幂。
    col_offsets = tl.arange(0, BLOCK_SIZE)
    # 例如 n_cols=781、BLOCK_SIZE=1024 时，781..1023 是 padding lanes。
    col_mask = col_offsets < n_cols

    # stride 是相邻两行起点之间相隔的元素数，不是字节数。
    input_row_ptr = input_ptr + row_idx * input_row_stride
    # padding lane 读成 -inf，因为 exp(-inf)=0，不会影响最大值或 Softmax 分母。
    row = tl.load(input_row_ptr + col_offsets, mask=col_mask, other=-float("inf"))

    # 先减最大值避免 exp 溢出；tl.max/tl.sum 都在当前 program 的 tile 内归约。
    # tl.max 的结果是当前整行的一个标量，并广播到每个 logical lane。
    row_minus_max = row - tl.max(row, axis=0)
    # numerator 仍是逐列向量；padding lane 的值为 0。
    numerator = tl.exp(row_minus_max)
    # denominator 是整行指数和标量，除法时广播到所有列。
    denominator = tl.sum(numerator, axis=0)
    softmax = numerator / denominator

    # 输出 stride 与输入分开传递，kernel 不假设两者的行距一定相同。
    output_row_ptr = output_ptr + row_idx * output_row_stride
    # padding 参与了片上计算，但 col_mask 阻止它写到输出边界之外。
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
    # 一个 program 要一次归约完整行，故 tile 向上补齐到二次幂而不是拆成多块。
    block_size = triton.next_power_of_2(n_cols)
    if block_size > 65_536:
        raise ValueError("this one-program-per-row lesson supports at most 65536 columns")

    # 输出与输入 shape/dtype/device 相同，但没有复制输入内容。
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
    # 默认 781 列不能整除 1024，可实际覆盖 padding mask 的教学路径。
    values = torch.randn(
        (args.rows, args.cols), device=device, dtype=torch.float32
    )
    actual = row_softmax(values, args.num_warps)
    # dim=1 表示 PyTorch 也沿每一行的列维度做 Softmax。
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
