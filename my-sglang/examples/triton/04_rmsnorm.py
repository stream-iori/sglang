"""Lesson 4: RMSNorm with a row reduction and fused weight multiplication."""

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
def rmsnorm_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """One program normalizes one row and applies the learned weight."""
    # 每个 program 对应一行 hidden state；列维就是模型的 hidden dimension。
    row_idx = tl.program_id(axis=0)
    # BLOCK_SIZE 向上补齐到二次幂，col_mask 标出真实 hidden dimensions。
    col_offsets = tl.arange(0, BLOCK_SIZE)
    col_mask = col_offsets < n_cols

    # 用行号乘 stride 找到这一行首地址，再为每个 logical lane 加列偏移。
    input_row_ptr = input_ptr + row_idx * input_row_stride
    # padding lane 读 0，使它的平方为 0，不会污染平方和。
    x = tl.load(input_row_ptr + col_offsets, mask=col_mask, other=0.0)
    # 即使输入以后改成 FP16/BF16，也用 FP32 计算平方和，减少归约误差。
    x_fp32 = x.to(tl.float32)
    # RMSNorm 不减均值：mean_square=(1/D) * sum(x_i^2)，D 必须用真实 n_cols。
    mean_square = tl.sum(x_fp32 * x_fp32, axis=0) / n_cols
    # rsqrt(a) 直接计算 1/sqrt(a)；eps 避免全零输入导致除零。
    inv_rms = tl.rsqrt(mean_square + eps)

    # weight 是所有行共享的一维可学习 gamma；每个 program 读取同一组列权重。
    weight = tl.load(weight_ptr + col_offsets, mask=col_mask, other=0.0)
    # inv_rms 是行级标量，会广播到每个元素：y_i=x_i*inv_rms*gamma_i。
    output = x_fp32 * inv_rms * weight
    output_row_ptr = output_ptr + row_idx * output_row_stride
    # 只将真实 hidden dimensions 转回 output dtype 并写回。
    tl.store(output_row_ptr + col_offsets, output, mask=col_mask)


def rmsnorm(
    values: torch.Tensor, weight: torch.Tensor, eps: float, num_warps: int
) -> torch.Tensor:
    validate_runtime_tensors(values, weight)
    if values.ndim != 2:
        raise ValueError("values must be a 2-D tensor")
    if values.shape[0] <= 0 or values.shape[1] <= 0:
        raise ValueError("values must have non-empty rows and columns")
    if weight.shape != (values.shape[1],):
        raise ValueError("weight must have shape (values.shape[1],)")
    if values.device != weight.device:
        raise ValueError("values and weight must be on the same device")
    if not values.is_contiguous() or not weight.is_contiguous():
        raise ValueError("values and weight must be contiguous")
    if eps <= 0:
        raise ValueError("eps must be positive")
    if num_warps <= 0:
        raise ValueError("num_warps must be positive")

    n_rows, n_cols = values.shape
    # 本教学 kernel 让一个 program 完整归约一行，所以不能把一行切给多个 program。
    block_size = triton.next_power_of_2(n_cols)
    if block_size > 65_536:
        raise ValueError("this one-program-per-row lesson supports at most 65536 columns")

    output = torch.empty_like(values)
    # grid 中每个 program 处理一行；weight 由所有 program 只读共享。
    rmsnorm_kernel[(n_rows,)](
        values,
        weight,
        output,
        values.stride(0),
        output.stride(0),
        n_cols,
        eps,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return output


def reference_rmsnorm(
    values: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    # keepdim=True 保留 [rows,1]，让每行的 inv_rms 沿 hidden dimension 广播。
    values_fp32 = values.float()
    inv_rms = torch.rsqrt(values_fp32.square().mean(dim=1, keepdim=True) + eps)
    return (values_fp32 * inv_rms * weight.float()).to(values.dtype)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1023)
    parser.add_argument("--cols", type=int, default=781)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--num-warps", type=int, default=8)
    args = parser.parse_args()

    if args.rows <= 0 or args.cols <= 0:
        raise ValueError("--rows and --cols must be positive")
    if args.eps <= 0:
        raise ValueError("--eps must be positive")
    if args.num_warps <= 0:
        raise ValueError("--num-warps must be positive")
    device = torch_device()

    torch.manual_seed(2)
    values = torch.randn(
        (args.rows, args.cols), device=device, dtype=torch.float32
    )
    weight = torch.randn(args.cols, device=device, dtype=torch.float32)
    actual = rmsnorm(values, weight, args.eps, args.num_warps)
    expected = reference_rmsnorm(values, weight, args.eps)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    synchronize()
    print(
        "PASS RMSNorm "
        f"(runtime={runtime_label()}, rows={args.rows}, cols={args.cols}, eps={args.eps}, "
        f"block_size={triton.next_power_of_2(args.cols)})"
    )


if __name__ == "__main__":
    main()
