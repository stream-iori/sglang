from __future__ import annotations

from my_sglang.cli import build_parser, main


def test_cli_parser_accepts_trace_and_prompt():
    # CLI 参数解析是轻量 smoke test，避免命令行入口的参数名被无意改坏。
    args = build_parser().parse_args(
        [
            "--input-ids",
            "1,2",
            "--token-ids",
            "10,11,12",
            "--max-new-tokens",
            "3",
            "--trace",
            "--overlap",
            "--chunked-prefill-size",
            "2",
            "--enable-radix-cache",
            "--page-size",
            "4",
            "--max-total-tokens",
            "128",
            "--max-prefill-tokens",
            "32",
        ]
    )

    assert args.input_ids == "1,2"
    assert args.runner == "scripted"
    assert args.token_ids == "10,11,12"
    assert args.max_new_tokens == 3
    assert args.trace is True
    assert args.overlap is True
    assert args.chunked_prefill_size == 2
    assert args.enable_radix_cache is True
    assert args.page_size == 4
    assert args.max_total_tokens == 128
    assert args.max_prefill_tokens == 32


def test_cli_runs_overlap_with_chunked_prefill(capsys):
    exit_code = main(
        [
            "--input-ids",
            "1,2,3,4,5",
            "--token-ids",
            "90,91,10,11,12",
            "--max-new-tokens",
            "2",
            "--overlap",
            "--chunked-prefill-size",
            "2",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == "10,11\n"
    assert captured.err == ""


def test_cli_documented_overlap_trace_order(capsys):
    exit_code = main(
        [
            "--input-ids",
            "1,2",
            "--token-ids",
            "10,11,12,13",
            "--max-new-tokens",
            "3",
            "--overlap",
            "--trace",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == "10,11,12\n"
    ordered_trace = [
        "forward:sample:B0:[10]",
        "forward:enqueue:forward+sample:B1",
        "forward:enqueue:forward+sample:B2",
        "event:sync:B1.copy_done",
        "forward:gather:B1:[10]",
        "forward:sample:B1:[11]",
        "copy:d2h:B1:[11]",
        "event:sync:B2.copy_done",
        "forward:gather:B2:[11]",
    ]
    positions = [captured.err.index(line) for line in ordered_trace]
    assert positions == sorted(positions)


def test_cli_tiny_runner_executes_real_model(capsys):
    exit_code = main(
        [
            "--input-ids",
            "1,2",
            "--runner",
            "tiny",
            "--max-new-tokens",
            "3",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == "94,94,94\n"
    assert captured.err == ""


def test_cli_validates_runner_specific_arguments(capsys):
    assert main(["--input-ids", "1", "--runner", "scripted"]) == 2
    captured = capsys.readouterr()
    assert "--token-ids is required" in captured.err

    assert main(["--input-ids", "1", "--runner", "tiny", "--overlap"]) == 2
    captured = capsys.readouterr()
    assert "supports only the synchronous scheduler" in captured.err

    assert main(["--input-ids", "256", "--runner", "tiny"]) == 2
    captured = capsys.readouterr()
    assert "outside vocab_size=256" in captured.err
