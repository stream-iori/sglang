"""Run a real Triton vector-add kernel with Triton's CPU interpreter."""

from __future__ import annotations

import argparse
import os

# This must be set before importing Triton. It makes kernel launches execute
# through the CPU interpreter instead of requiring a CUDA/ROCm device.
os.environ.setdefault("TRITON_INTERPRET", "1")

import torch
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """One program handles BLOCK_SIZE consecutive vector elements."""
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(output_ptr + offsets, x + y, mask=mask)


def is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1003)
    parser.add_argument("--block-size", type=int, default=256)
    args = parser.parse_args()

    if args.n < 0:
        raise ValueError("--n must be non-negative")
    if not is_power_of_two(args.block_size):
        raise ValueError("--block-size must be a positive power of two")

    x = torch.arange(args.n, dtype=torch.float32)
    y = torch.arange(args.n, 0, -1, dtype=torch.float32)
    output = torch.empty_like(x)
    grid = (triton.cdiv(args.n, args.block_size),)
    vector_add_kernel[grid](x, y, output, args.n, BLOCK_SIZE=args.block_size)

    torch.testing.assert_close(output, x + y)
    print(
        "PASS real @triton.jit kernel via CPU interpreter "
        f"(n={args.n}, block_size={args.block_size}, grid={grid[0]})"
    )


if __name__ == "__main__":
    main()
