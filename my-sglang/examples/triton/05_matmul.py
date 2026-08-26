"""Lesson 5: tiled matrix multiplication with a 2-D Triton launch grid."""

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
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """One program computes one ``BLOCK_SIZE_M × BLOCK_SIZE_N`` output tile."""
    # 二维 grid 中，axis=0 选择 C 的行 tile，axis=1 选择 C 的列 tile。
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # offs_m/offs_n 是这个 program 覆盖的 C 全局行列；offs_k 是 K tile 内坐标。
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    # 每个 C 元素都要沿完整 K 维求和；FP32 accumulator 跨 K tiles 保存部分和。
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # 沿 K 维逐 tile 前进；每轮加载 A 的竖块和 B 的横块并累加点积。
    for k_tile in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_start = k_tile * BLOCK_SIZE_K
        # 广播形成 [BLOCK_M, BLOCK_K] 指针矩阵：每一行来自 A 的一个输出行。
        # stride_am/stride_ak 的单位是元素，可支持非固定的物理行距。
        a_ptrs = (
            a_ptr
            + offs_m[:, None] * stride_am
            + (k_start + offs_k[None, :]) * stride_ak
        )
        # 广播形成 [BLOCK_K, BLOCK_N] 指针矩阵：每一列贡献给 C 的一个输出列。
        b_ptrs = (
            b_ptr
            + (k_start + offs_k[:, None]) * stride_bk
            + offs_n[None, :] * stride_bn
        )
        # M/N/K 都可能不能整除 tile；越界输入必须独立按各自 shape 屏蔽。
        a_mask = (offs_m[:, None] < M) & (k_start + offs_k[None, :] < K)
        b_mask = (k_start + offs_k[:, None] < K) & (offs_n[None, :] < N)
        # 越界位置补 0 是因为 0 对点积求和没有贡献。
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        # 等价于 accumulator += a @ b；输出 shape 是 [BLOCK_M, BLOCK_N]。
        accumulator = tl.dot(a, b, accumulator)

    # K tiles 全部累加完成后，才为当前 C tile 计算一次输出地址并写回。
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    # 最右/最下输出 tile 可能越过真实 M×N，只写有效的 C 元素。
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
) -> torch.Tensor:
    validate_runtime_tensors(a, b)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("a and b must be 2-D tensors")
    if a.device != b.device or a.dtype != b.dtype:
        raise ValueError("a and b must have the same device and dtype")
    if a.shape[1] != b.shape[0]:
        raise ValueError("a.shape[1] must equal b.shape[0]")
    if min(a.shape[0], a.shape[1], b.shape[1]) <= 0:
        raise ValueError("matrix dimensions must be positive")
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("a and b must be contiguous")
    if not all(is_power_of_two(v) for v in (block_m, block_n, block_k)):
        raise ValueError("block sizes must be positive powers of two")
    if num_warps <= 0:
        raise ValueError("num_warps must be positive")

    M, K = a.shape
    _, N = b.shape
    # 数学 shape 为 [M,K] @ [K,N] -> [M,N]。
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    # 二维 grid：axis=0 选择输出的行 tile，axis=1 选择输出的列 tile。
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    # 把每一维 stride 显式传入 kernel，设备代码无需假定固定的地址公式。
    matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=block_k,
        num_warps=num_warps,
    )
    return c


def is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=257)
    parser.add_argument("--n", type=int, default=263)
    parser.add_argument("--k", type=int, default=269)
    parser.add_argument("--block-m", type=int, default=32)
    parser.add_argument("--block-n", type=int, default=32)
    parser.add_argument("--block-k", type=int, default=32)
    parser.add_argument("--num-warps", type=int, default=4)
    args = parser.parse_args()

    if args.m <= 0 or args.n <= 0 or args.k <= 0:
        raise ValueError("--m, --n, and --k must be positive")
    if not all(
        is_power_of_two(value)
        for value in (args.block_m, args.block_n, args.block_k)
    ):
        raise ValueError("all block sizes must be positive powers of two")
    if args.num_warps <= 0:
        raise ValueError("--num-warps must be positive")
    device = torch_device()

    torch.manual_seed(3)
    # 默认 257/263/269 都不能整除 32，可同时验证 M、N、K 三个方向的尾 tile mask。
    a = torch.randn((args.m, args.k), device=device, dtype=torch.float16)
    b = torch.randn((args.k, args.n), device=device, dtype=torch.float16)
    actual = matmul(
        a,
        b,
        args.block_m,
        args.block_n,
        args.block_k,
        args.num_warps,
    )
    # torch.matmul 提供独立 reference；FP16 输出允许与 FP32 累加路径有小舍入差异。
    expected = torch.matmul(a, b)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    synchronize()
    print(
        "PASS tiled matmul "
        f"(runtime={runtime_label()}, M={args.m}, N={args.n}, K={args.k}, "
        f"tile={args.block_m}x{args.block_n}x{args.block_k})"
    )


if __name__ == "__main__":
    main()
