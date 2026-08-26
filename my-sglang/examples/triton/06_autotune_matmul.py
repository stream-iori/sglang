"""Lesson 6: let Triton autotune several real matrix-multiplication configs."""

from __future__ import annotations

import argparse

import torch
import triton
import triton.language as tl

from runtime_utils import (
    interpreter_enabled,
    require_benchmark_runtime,
    runtime_label,
    synchronize,
    torch_device,
    validate_runtime_tensors,
)


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32},
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_warps=8,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def autotuned_matmul_kernel(
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
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k_tile in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_start = k_tile * BLOCK_SIZE_K
        a_ptrs = (
            a_ptr
            + offs_m[:, None] * stride_am
            + (k_start + offs_k[None, :]) * stride_ak
        )
        b_ptrs = (
            b_ptr
            + (k_start + offs_k[:, None]) * stride_bk
            + offs_n[None, :] * stride_bn
        )
        a_mask = (offs_m[:, None] < M) & (k_start + offs_k[None, :] < K)
        b_mask = (k_start + offs_k[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        accumulator = tl.dot(a, b, accumulator)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def autotuned_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    validate_runtime_tensors(a, b)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("expected a[M, K] and b[K, N]")
    if a.device != b.device or a.dtype != b.dtype:
        raise ValueError("a and b must have the same device and dtype")
    if min(a.shape[0], a.shape[1], b.shape[1]) <= 0:
        raise ValueError("matrix dimensions must be positive")
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("a and b must be contiguous")

    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    # grid 是 callable：autotuner 为每个候选配置传入不同的 meta-parameters。
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE_M"]),
        triton.cdiv(N, meta["BLOCK_SIZE_N"]),
    )
    kernel_args = (
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
    )
    if interpreter_enabled():
        # CPU interpreter 没有 GPU benchmark driver，无法对候选配置计时。
        # 因此绕过 Autotuner，固定用第一组配置验证 kernel 数值语义。
        config = autotuned_matmul_kernel.configs[0]
        fixed_grid = grid(config.kwargs)
        autotuned_matmul_kernel.fn[fixed_grid](
            *kernel_args,
            **config.kwargs,
            num_warps=config.num_warps,
        )
    else:
        # CUDA 模式会对 configs 逐个 benchmark，并缓存最快配置。
        autotuned_matmul_kernel[grid](*kernel_args)
    return c


def benchmark(a: torch.Tensor, b: torch.Tensor) -> None:
    require_benchmark_runtime()
    M, K = a.shape
    _, N = b.shape
    triton_ms = triton.testing.do_bench(lambda: autotuned_matmul(a, b))
    torch_ms = triton.testing.do_bench(lambda: torch.matmul(a, b))
    operation_count = 2 * M * N * K
    triton_tflops = operation_count * 1e-12 / (triton_ms * 1e-3)
    torch_tflops = operation_count * 1e-12 / (torch_ms * 1e-3)
    print("provider  ms        TFLOPS")
    print(f"triton    {triton_ms:8.4f}  {triton_tflops:8.2f}")
    print(f"torch     {torch_ms:8.4f}  {torch_tflops:8.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()

    if args.m <= 0 or args.n <= 0 or args.k <= 0:
        raise ValueError("--m, --n, and --k must be positive")
    device = torch_device()

    torch.manual_seed(4)
    a = torch.randn((args.m, args.k), device=device, dtype=torch.float16)
    b = torch.randn((args.k, args.n), device=device, dtype=torch.float16)
    # 第一次调用会实际编译并测量多个候选配置；相同 M/N/K 后续复用选择结果。
    actual = autotuned_matmul(a, b)
    expected = torch.matmul(a, b)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    synchronize()
    print(
        "PASS autotuned matmul "
        f"(runtime={runtime_label()}, M={args.m}, N={args.n}, K={args.k})"
    )

    if args.benchmark:
        benchmark(a, b)


if __name__ == "__main__":
    main()
