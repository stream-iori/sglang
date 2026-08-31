"""Lesson 2: fuse ``silu(x + bias)`` in one Triton kernel."""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from runtime_utils import (
    require_benchmark_runtime,
    runtime_label,
    synchronize,
    torch_device,
    validate_runtime_tensors,
)


@triton.jit
def fused_silu_kernel(x_ptr, bias_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """One program loads a 1-D tile, performs add + SiLU, then stores once."""
    # 与 01 相同，pid 选择当前 program 负责的连续一维 tile。
    pid = tl.program_id(axis=0)
    # 例如 pid=3、BLOCK_SIZE=256 时，当前 program 尝试处理 768..1023。
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # n_elements 不一定是 BLOCK_SIZE 的整数倍，尾 tile 的越界位置必须屏蔽。
    mask = offsets < n_elements

    # 两次 load 从 HBM 读取相同位置的 x 和 bias；无效 tile 位置补 0，且不会写回。
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
    # z 是融合后的中间值，只保留在当前 program 内，不分配与 z 等大的 HBM tensor。
    z = x + bias
    # SiLU(z) = z * sigmoid(z)。z 只存在于当前 program 内，不写中间 HBM 数组。
    output = z * tl.sigmoid(z)
    # 只把最终激活写回一次；mask 同时保护 output 的尾部边界。
    tl.store(output_ptr + offsets, output, mask=mask)


def fused_silu(
    x: torch.Tensor, bias: torch.Tensor, block_size: int, num_warps: int
) -> torch.Tensor:
    """Validate inputs, allocate output, and launch the Triton kernel."""
    validate_runtime_tensors(x, bias)
    if x.device != bias.device or x.dtype != bias.dtype:
        raise ValueError("x and bias must have the same device and dtype")
    if x.shape != bias.shape:
        raise ValueError("x and bias must have the same shape")
    if x.numel() <= 0:
        raise ValueError("x and bias must be non-empty")
    if not x.is_contiguous() or not bias.is_contiguous():
        raise ValueError("x and bias must be contiguous")
    if not is_power_of_two(block_size) or num_warps <= 0:
        raise ValueError("block_size must be a power of two and num_warps positive")

    # contiguous 保证把任意输入 shape 展平成一维后，offset 仍对应连续元素。
    # empty_like 只分配最终输出；kernel 不需要 z 的额外缓冲区。
    output = torch.empty_like(x)
    n_elements = x.numel()
    # ceil-div 保证最后不足一个 BLOCK_SIZE 的元素也有 program 覆盖。
    grid = (triton.cdiv(n_elements, block_size),)
    # block_size/num_warps 控制 tile 和底层并行映射，不改变 SiLU 的数学结果。
    fused_silu_kernel[grid](
        x,
        bias,
        output,
        n_elements,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return output


def benchmark(n: int, block_size: int, num_warps: int) -> None:
    """Compare stable execution time and effective bandwidth, excluding JIT warmup."""
    require_benchmark_runtime()
    x = torch.randn(n, device=torch_device(), dtype=torch.float32)
    bias = torch.randn_like(x)

    # do_bench 会 warmup、重复执行并处理 CUDA 同步；结果单位是毫秒。
    triton_ms = triton.testing.do_bench(
        lambda: fused_silu(x, bias, block_size, num_warps)
    )
    torch_ms = triton.testing.do_bench(lambda: F.silu(x + bias))

    # 按算法最少流量计算有效带宽：读取 x、bias，写 output，共 3 个 tensor。
    logical_gb = 3 * n * x.element_size() * 1e-9
    triton_gbps = logical_gb / (triton_ms * 1e-3)
    torch_gbps = logical_gb / (torch_ms * 1e-3)
    print("provider  ms        effective_GB/s")
    print(f"triton    {triton_ms:8.4f}  {triton_gbps:14.2f}")
    print(f"torch     {torch_ms:8.4f}  {torch_gbps:14.2f}")


def is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1_000_003)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()

    if args.n <= 0:
        raise ValueError("--n must be positive")
    if not is_power_of_two(args.block_size):
        raise ValueError("--block-size must be a positive power of two")
    if args.num_warps <= 0:
        raise ValueError("--num-warps must be positive")
    device = torch_device()

    torch.manual_seed(0)
    # 在 interpreter 模式这些是 CPU tensor；CUDA 模式则直接创建 GPU tensor。
    x = torch.randn(args.n, device=device, dtype=torch.float32)
    bias = torch.randn_like(x)
    actual = fused_silu(x, bias, args.block_size, args.num_warps)
    # PyTorch 分开执行 add 与 SiLU，作为独立于 Triton kernel 的数值 reference。
    expected = F.silu(x + bias)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    synchronize()
    print(
        "PASS fused SiLU "
        f"(runtime={runtime_label()}, n={args.n}, "
        f"block_size={args.block_size}, num_warps={args.num_warps})"
    )

    if args.benchmark:
        benchmark(args.n, args.block_size, args.num_warps)


if __name__ == "__main__":
    main()
