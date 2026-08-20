from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = PROJECT_ROOT / "examples" / "triton"


def run_example(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(EXAMPLES / script), *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    ("script", "args", "pass_marker"),
    [
        ("01_vector_add.py", ("--n", "1", "--block-size", "256"), "PASS vector_add"),
        ("01_vector_add.py", ("--n", "1003", "--block-size", "256"), "PASS vector_add"),
        (
            "02_vector_fusion.py",
            ("--n", "257", "--block-size", "256", "--scale", "-0.75"),
            "PASS vector_fusion",
        ),
        (
            "03_row_softmax.py",
            ("--rows", "3", "--cols", "257", "--block-size", "512"),
            "PASS row_softmax",
        ),
        (
            "04_rmsnorm.py",
            ("--rows", "3", "--cols", "257", "--block-size", "512"),
            "PASS rmsnorm",
        ),
    ],
)
def test_cpu_triton_examples_match_their_numpy_references(
    script: str, args: tuple[str, ...], pass_marker: str
):
    result = run_example(script, *args)
    assert result.returncode == 0, result.stderr
    assert pass_marker in result.stdout


def test_vector_add_trace_shows_tail_program_mask():
    result = run_example("01_vector_add.py", "--n", "1003", "--block-size", "256", "--trace")
    assert result.returncode == 0, result.stderr
    assert "pid=3 offsets=768..1023 valid=768..1002" in result.stdout


@pytest.mark.parametrize(
    ("script", "args", "error"),
    [
        ("01_vector_add.py", ("--block-size", "0"), "block_size must be > 0"),
        (
            "03_row_softmax.py",
            ("--cols", "257", "--block-size", "256"),
            "block_size must be >= cols",
        ),
        ("04_rmsnorm.py", ("--eps", "0"), "eps must be > 0"),
    ],
)
def test_cpu_triton_examples_reject_invalid_lesson_inputs(
    script: str, args: tuple[str, ...], error: str
):
    result = run_example(script, *args)
    assert result.returncode != 0
    assert error in result.stderr
