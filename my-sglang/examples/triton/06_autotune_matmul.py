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
    # 每个 config 描述一个候选输出 tile、K tile 和执行该 program 的 warp 数。
    # 候选过大可能增加寄存器压力，过小又可能无法充分复用数据或利用 GPU。
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
    # M/N/K 改变时重新选择；相同 key 的后续调用复用已缓存的最快配置。
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
    """Compute the same tiled C=A@B as lesson 5, with autotuned meta-parameters."""
    # axis=0/1 分别选择输出 C 的行/列 tile；config 决定每个 tile 的大小。
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    # offs_m/offs_n 是 C 的全局坐标，offs_k 是当前 K 分块内的逻辑坐标。
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    # 无论选择哪个 config，数学结果都是沿 K 维累加得到同一个 [M,N] 矩阵。
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # K 是 reduction dimension；最后一个 K tile 不足时由 a_mask/b_mask 补零。
    for k_tile in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_start = k_tile * BLOCK_SIZE_K
        # 通过二维广播构造 A[BLOCK_M,BLOCK_K] 和 B[BLOCK_K,BLOCK_N] 的地址。
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
        # 三个维度分别做边界保护，使任意正 M/N/K 都不要求整除候选 tile。
        a_mask = (offs_m[:, None] < M) & (k_start + offs_k[None, :] < K)
        b_mask = (k_start + offs_k[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        # 对当前 K tile 做矩阵乘加，FP32 accumulator 保留跨 tile 部分和。
        accumulator = tl.dot(a, b, accumulator)

    # 只在 K reduction 完成后写一次 C；尾部 program 用 c_mask 丢弃越界元素。
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
    # 输出 shape 固定为 [M,N]，autotuner 只改变计算它的 tile 方案。
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
    # 每个 C 元素包含 K 次乘法和 K 次加法，按惯例近似计作 2*M*N*K FLOPs。
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
    # 默认方阵便于观察 autotune；也可传非整除 shape 验证各候选的边界 mask。
    a = torch.randn((args.m, args.k), device=device, dtype=torch.float16)
    b = torch.randn((args.k, args.n), device=device, dtype=torch.float16)
    # 第一次调用会实际编译并测量多个候选配置；相同 M/N/K 后续复用选择结果。
    actual = autotuned_matmul(a, b)
    # Autotune 选择必须只影响性能，所有候选的结果都应与同一 PyTorch reference 一致。
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
