from __future__ import annotations

from my_sglang.cli import build_parser


def test_cli_parser_accepts_trace_and_prompt():
    # CLI 参数解析是轻量 smoke test，避免命令行入口的参数名被无意改坏。
    args = build_parser().parse_args(
        [
            "--prompt",
            "Hello",
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

    assert args.prompt == "Hello"
    assert args.max_new_tokens == 3
    assert args.trace is True
    assert args.overlap is True
    assert args.chunked_prefill_size == 2
    assert args.enable_radix_cache is True
    assert args.page_size == 4
    assert args.max_total_tokens == 128
    assert args.max_prefill_tokens == 32
