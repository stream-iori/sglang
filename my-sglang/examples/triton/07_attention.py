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
    # 一维 grid 展平了 [batch, n_heads]；每个 program 负责一个 query head。
    program_id = tl.program_id(axis=0)
    # 例如 n_heads=4、program_id=6 对应 batch_idx=1、head_idx=2。
    batch_idx = program_id // n_heads
    head_idx = program_id % n_heads
    # BLOCK_D 向上补齐到二次幂；d_mask 屏蔽超出真实 head_dim 的 lanes。
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < head_dim

    # Q 的 shape 是 [B,H,D]。固定 batch/head 后，q_ptrs 选择长度 D 的向量。
    q_ptrs = (
        q_ptr
        + batch_idx * stride_qb
        + head_idx * stride_qh
        + offs_d * stride_qd
    )
    # query 会与每个 context key 点积，因此只加载一次并在循环中复用。
    query = tl.load(q_ptrs, mask=d_mask, other=0.0).to(tl.float32)

    # Online softmax 状态：m 是目前最大 score，l 是归一化分母，acc 是加权 V。
    # 三者都以 running_max 为共同指数基准，跨 context tiles 递推更新。
    running_max = -float("inf")
    running_sum = 0.0
    accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)

    # 不一次物化全部 n_ctx 个 scores，而是每轮流式处理 BLOCK_N 个 K/V 位置。
    for n_start in tl.range(0, n_ctx, BLOCK_N):
        # offs_n 是当前 tile 的全局 context positions；最后一块可能包含 padding。
        offs_n = n_start + tl.arange(0, BLOCK_N)
        n_mask = offs_n < n_ctx

        # K 的 shape 是 [B,H,N,D]；广播得到 [BLOCK_N,BLOCK_D] 指针矩阵。
        k_ptrs = (
            k_ptr
            + batch_idx * stride_kb
            + head_idx * stride_kh
            + offs_n[:, None] * stride_kn
            + offs_d[None, :] * stride_kd
        )
        # context 与 head-dim 两个方向都要 mask；无效 D lane 补 0，不影响点积。
        kv_mask = n_mask[:, None] & d_mask[None, :]
        keys = tl.load(k_ptrs, mask=kv_mask, other=0.0).to(tl.float32)
        # 沿 D 维归约得到当前 tile 的 [BLOCK_N] scores，再乘 1/sqrt(D)。
        scores = tl.sum(keys * query[None, :], axis=1) * scale
        # padding context position 必须是 -inf，使 exp 后权重严格为 0。
        scores = tl.where(n_mask, scores, -float("inf"))

        # 合并“旧 tile 的 softmax 状态”和“当前 tile”的数值稳定公式。
        # new_max 是到目前为止所有有效 score 的最大值，也是新的指数基准。
        tile_max = tl.max(scores, axis=0)
        new_max = tl.maximum(running_max, tile_max)
        # 最大值变大后，旧的分母与加权和都要乘 exp(old_max-new_max) 重标定。
        old_scale = tl.exp(running_max - new_max)
        # 这里的 probabilities 尚未除以最终分母，是基于 new_max 的未归一化权重。
        probabilities = tl.exp(scores - new_max)
        new_sum = running_sum * old_scale + tl.sum(probabilities, axis=0)

        # V 与 K 具有相同 [B,H,N,D] 布局，使用同一 context/head-dim mask。
        v_ptrs = (
            v_ptr
            + batch_idx * stride_vb
            + head_idx * stride_vh
            + offs_n[:, None] * stride_vn
            + offs_d[None, :] * stride_vd
        )
        values = tl.load(v_ptrs, mask=kv_mask, other=0.0).to(tl.float32)
        # 加权 V 沿 N 维归约成 [BLOCK_D]；旧 accumulator 使用相同 old_scale。
        accumulator = accumulator * old_scale + tl.sum(
            probabilities[:, None] * values, axis=0
        )
        # 保存合并后的 (m,l,acc)，供下一个 context tile 使用。
        running_max = new_max
        running_sum = new_sum

    # 全部 context 处理后才除以 Softmax 分母，得到这个 query head 的输出向量。
    output = accumulator / running_sum
    # 输出 shape 是 [B,H,D]，当前 program 只写自己的 batch/head 行。
    output_ptrs = (
        output_ptr
        + batch_idx * stride_ob
        + head_idx * stride_oh
        + offs_d * stride_od
    )
    # BLOCK_D 的 padding lane 不属于真实输出，必须用 d_mask 丢弃。
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
    # 每个 [batch,head] query 产生一个长度 head_dim 的输出。
    output = torch.empty_like(query)
    # 一维 grid 展平 B×H；kernel 内再用整除和取模恢复两个坐标。
    grid = (batch * n_heads,)
    # Scaled dot-product attention 使用 1/sqrt(D) 防止 D 增大时 score 过大。
    scale = 1.0 / math.sqrt(head_dim)
    # stride 全部显式传入，kernel 的地址公式与具体 tensor 布局参数保持对应。
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
    # 直接物化完整 scores/probabilities，写法简单，适合作为 online 算法的 reference。
    scale = 1.0 / math.sqrt(query.shape[-1])
    # [B,H,D] · [B,H,N,D] -> [B,H,N]，D 是点积归约维。
    scores = torch.einsum("bhd,bhnd->bhn", query.float(), key.float()) * scale
    # 对 context N 维归一化，再与 V 加权求和回到 [B,H,D]。
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
    # context=257 默认会产生一个不足 BLOCK_N 的尾 tile，用于覆盖 n_mask 路径。
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
    # reference 一次计算完整 Softmax，用来验证分 tile online recurrence 等价。
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
