"""Lesson 1: Triton vector add in CUDA or CPU-interpreter mode."""

from __future__ import annotations

import argparse

import torch
import triton
import triton.language as tl

from runtime_utils import runtime_label, synchronize, torch_device


@triton.jit
def vector_add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """一个 program instance 负责连续的 ``BLOCK_SIZE`` 个元素。"""
    # 与 CUDA 的 blockIdx.x 最接近：它标识当前 program，而非单个 thread。
    program_id = tl.program_id(axis=0)
    # tl.arange 产生逻辑 lane。它描述一个 tile，不等价于创建 BLOCK_SIZE 个 CUDA threads。
    # 例如 program_id=3、BLOCK_SIZE=256 时，offsets 是 768..1023。
    offsets = program_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # 最后一个 tile 往往超过向量末尾；mask=False 的 lane 禁止访问显存。
    mask = offsets < n_elements
    # 指针加 offsets 是逐 lane 的 global-memory 访问。other 只供 mask=False 的 lane 使用，
    # 防止无效读参与后续计算；真正的结果不会由这些 lane 写回。
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    # 一个 kernel 内完成 load -> add -> store，没有 x + y 的中间 HBM 数组。
    tl.store(output_ptr + offsets, x + y, mask=mask)


def is_power_of_two(value: int) -> bool:
    # 本入门 kernel 选择 2 的幂 tile，符合 tl.arange 常见的编译布局要求。
    return value > 0 and value & (value - 1) == 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1003)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--num-warps", type=int, default=4)
    args = parser.parse_args()

    if args.n <= 0:
        raise ValueError("--n must be positive")
    if not is_power_of_two(args.block_size):
        raise ValueError("--block-size must be a positive power of two")
    if args.num_warps <= 0:
        raise ValueError("--num-warps must be positive")
    device = torch_device()

    # CUDA 模式创建 GPU tensor；interpreter 模式创建 CPU tensor，由 Triton 逐 op 解释执行。
    x = torch.arange(args.n, dtype=torch.float32, device=device)
    # y 的起点是 n、终点不包含 0、步长为 -1，因此 y=[n, n-1, ..., 1]。
    y = torch.arange(args.n, 0, -1, dtype=torch.float32, device=device)
    # empty_like 分配与 x 同 shape/dtype/device 的输出缓冲区，但不拷贝 x 的值。
    # 它的初始内容未定义，kernel 的 masked store 负责填充有效位置。
    output = torch.empty_like(x)
    # grid 决定启动多少个 program。ceil_div 保证尾部不足一个 tile 的元素也被覆盖。
    grid = (triton.cdiv(args.n, args.block_size),)
    # BLOCK_SIZE 是编译期常量，编译器据此生成 tl.arange 和布局。
    # num_warps 是 launch/编译提示：影响底层并行实现，不改变 x + y 的数学语义。
    vector_add_kernel[grid](
        x, y, output, args.n, BLOCK_SIZE=args.block_size, num_warps=args.num_warps
    )
    # PyTorch 的 x + y 是同一 device 上的 reference。assert_close 会比较每个元素，
    # 用于检查 Triton 的数值结果；它不是 Triton 与 PyTorch 的性能比较。
    torch.testing.assert_close(output, x + y)
    synchronize()
    print(
        "PASS Triton vector-add kernel "
        f"(runtime={runtime_label()}, n={args.n}, block_size={args.block_size}, "
        f"grid={grid[0]}, num_warps={args.num_warps})"
    )


if __name__ == "__main__":
    main()
