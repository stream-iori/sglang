"""Shared runtime helpers for real CUDA and Triton's CPU interpreter."""

from __future__ import annotations

import os

import torch


def interpreter_enabled() -> bool:
    """TRITON_INTERPRET must be set before importing Triton in the process."""
    return os.environ.get("TRITON_INTERPRET") == "1"


def torch_device() -> torch.device:
    if interpreter_enabled():
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU is available. Set TRITON_INTERPRET=1 before starting "
            "Python to run with Triton's CPU interpreter."
        )
    return torch.device("cuda")


def validate_runtime_tensors(*tensors: torch.Tensor) -> None:
    expected_type = "cpu" if interpreter_enabled() else "cuda"
    if any(tensor.device.type != expected_type for tensor in tensors):
        raise ValueError(
            f"all tensors must be on {expected_type} for the active Triton runtime"
        )


def synchronize() -> None:
    # Interpreter calls are synchronous CPU work; CUDA launches need an explicit barrier.
    if not interpreter_enabled():
        torch.cuda.synchronize()


def runtime_label() -> str:
    return "cpu-interpreter" if interpreter_enabled() else "cuda"


def require_benchmark_runtime() -> None:
    if interpreter_enabled():
        raise RuntimeError(
            "benchmark is disabled in TRITON_INTERPRET mode; interpreter timings "
            "do not represent GPU performance"
        )
