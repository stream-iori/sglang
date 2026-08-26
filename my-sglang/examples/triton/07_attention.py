"""Lesson 7: single-query attention with online softmax over context tiles.

This is an educational Triton kernel, not a production FlashAttention replacement.
"""

from __future__ import annotations

import argparse
import math

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
def single_query_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    output_ptr,
    n_heads,
    n_ctx,
    head_dim,
    scale,
    stride_qb,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_od,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """One program computes attention for one ``(batch, head)`` query."""
    program_id = tl.program_id(axis=0)
    batch_idx = program_id // n_heads
    head_idx = program_id % n_heads
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < head_dim

    q_ptrs = (
        q_ptr
        + batch_idx * stride_qb
        + head_idx * stride_qh
        + offs_d * stride_qd
    )
    query = tl.load(q_ptrs, mask=d_mask, other=0.0).to(tl.float32)

    # Online softmax 状态：m 是目前最大 score，l 是归一化分母，acc 是加权 V。
    running_max = -float("inf")
    running_sum = 0.0
    accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for n_start in tl.range(0, n_ctx, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        n_mask = offs_n < n_ctx

        k_ptrs = (
            k_ptr
            + batch_idx * stride_kb
            + head_idx * stride_kh
            + offs_n[:, None] * stride_kn
            + offs_d[None, :] * stride_kd
        )
        kv_mask = n_mask[:, None] & d_mask[None, :]
        keys = tl.load(k_ptrs, mask=kv_mask, other=0.0).to(tl.float32)
        scores = tl.sum(keys * query[None, :], axis=1) * scale
        scores = tl.where(n_mask, scores, -float("inf"))

        # 合并“旧 tile 的 softmax 状态”和“当前 tile”的数值稳定公式。
        tile_max = tl.max(scores, axis=0)
        new_max = tl.maximum(running_max, tile_max)
        old_scale = tl.exp(running_max - new_max)
        probabilities = tl.exp(scores - new_max)
        new_sum = running_sum * old_scale + tl.sum(probabilities, axis=0)

        v_ptrs = (
            v_ptr
            + batch_idx * stride_vb
            + head_idx * stride_vh
            + offs_n[:, None] * stride_vn
            + offs_d[None, :] * stride_vd
        )
        values = tl.load(v_ptrs, mask=kv_mask, other=0.0).to(tl.float32)
        accumulator = accumulator * old_scale + tl.sum(
            probabilities[:, None] * values, axis=0
        )
        running_max = new_max
        running_sum = new_sum

    output = accumulator / running_sum
    output_ptrs = (
        output_ptr
        + batch_idx * stride_ob
        + head_idx * stride_oh
        + offs_d * stride_od
    )
    tl.store(output_ptrs, output, mask=d_mask)


def single_query_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_n: int,
    num_warps: int,
) -> torch.Tensor:
    """Launch attention for Q[B,H,D] and K/V[B,H,N,D]."""
    validate_runtime_tensors(query, key, value)
    if query.ndim != 3 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("expected Q[B,H,D] and K/V[B,H,N,D]")
    if key.shape != value.shape:
        raise ValueError("key and value must have the same shape")
    if query.device != key.device or query.device != value.device:
        raise ValueError("query, key, and value must be on the same device")
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise ValueError("query, key, and value must have the same dtype")
    batch, n_heads, head_dim = query.shape
    if key.shape[:2] != (batch, n_heads) or key.shape[3] != head_dim:
        raise ValueError("query, key, and value dimensions do not match")
    if key.shape[2] <= 0 or head_dim <= 0:
        raise ValueError("context length and head dimension must be positive")
    if not query.is_contiguous() or not key.is_contiguous() or not value.is_contiguous():
        raise ValueError("query, key, and value must be contiguous")
    if not is_power_of_two(block_n) or num_warps <= 0:
        raise ValueError("block_n must be a power of two and num_warps positive")

    block_d = triton.next_power_of_2(head_dim)
    if block_d > 128:
        raise ValueError("this educational kernel supports head_dim <= 128")

    n_ctx = key.shape[2]
    output = torch.empty_like(query)
    grid = (batch * n_heads,)
    scale = 1.0 / math.sqrt(head_dim)
    single_query_attention_kernel[grid](
        query,
        key,
        value,
        output,
        n_heads,
        n_ctx,
        head_dim,
        scale,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        key.stride(3),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        value.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=num_warps,
    )
    return output


def reference_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> torch.Tensor:
    scale = 1.0 / math.sqrt(query.shape[-1])
    scores = torch.einsum("bhd,bhnd->bhn", query.float(), key.float()) * scale
    probabilities = torch.softmax(scores, dim=-1)
    return torch.einsum("bhn,bhnd->bhd", probabilities, value.float()).to(
        query.dtype
    )


def is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--context", type=int, default=257)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--block-n", type=int, default=32)
    parser.add_argument("--num-warps", type=int, default=4)
    args = parser.parse_args()

    if min(args.batch, args.heads, args.context, args.head_dim) <= 0:
        raise ValueError("batch, heads, context, and head-dim must be positive")
    if not is_power_of_two(args.block_n):
        raise ValueError("--block-n must be a positive power of two")
    if args.num_warps <= 0:
        raise ValueError("--num-warps must be positive")
    device = torch_device()

    torch.manual_seed(5)
    query = torch.randn(
        (args.batch, args.heads, args.head_dim),
        device=device,
        dtype=torch.float32,
    )
    key = torch.randn(
        (args.batch, args.heads, args.context, args.head_dim),
        device=device,
        dtype=torch.float32,
    )
    value = torch.randn_like(key)
    actual = single_query_attention(
        query, key, value, args.block_n, args.num_warps
    )
    expected = reference_attention(query, key, value)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    synchronize()
    print(
        "PASS online-softmax attention "
        f"(runtime={runtime_label()}, batch={args.batch}, heads={args.heads}, "
        f"context={args.context}, "
        f"head_dim={args.head_dim}, block_n={args.block_n})"
    )


if __name__ == "__main__":
    main()
