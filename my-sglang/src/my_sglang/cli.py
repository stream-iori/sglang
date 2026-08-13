"""运行单个请求的命令行学习入口。"""

from __future__ import annotations

import argparse
import sys

from my_sglang.models import Req, SamplingParams
from my_sglang.overlap_scheduler import MiniOverlapScheduler
from my_sglang.runner import FakeCudaRunner
from my_sglang.scheduler import MiniScheduler


def build_parser() -> argparse.ArgumentParser:
    """声明 CLI 参数；独立函数便于测试解析行为。"""
    parser = argparse.ArgumentParser(description="运行 CPU Fake CUDA 的迷你 SGLang 调度器。")
    parser.add_argument("--input-ids", required=True, help="逗号分隔的 prompt token ids，例如 1,2")
    parser.add_argument("--token-ids", required=True, help="逗号分隔的确定性采样 token ids")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--overlap", action="store_true")
    parser.add_argument("--enable-radix-cache", action="store_true")
    parser.add_argument("--page-size", type=int, default=1)
    parser.add_argument("--max-total-tokens", type=int, default=8192)
    parser.add_argument("--max-prefill-tokens", type=int, default=-1)
    parser.add_argument(
        "--chunked-prefill-size",
        type=int,
        default=-1,
        help="每轮最多 prefill 多少个 prompt token；<=0 表示关闭。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """完成 tokenize -> 调度生成 -> decode 文本的最小闭环。"""

    # argv=None 时读取真实命令行；测试可传入自定义参数列表。
    args = build_parser().parse_args(argv)
    input_ids = [int(value) for value in args.input_ids.split(",") if value]
    token_ids = [int(value) for value in args.token_ids.split(",") if value]
    if not input_ids:
        print("prompt produced no input tokens", file=sys.stderr)
        return 2
    # 阶段 2：CPU Fake CUDA runner 保留 FutureMap/stream/event 调度语义。
    runner = FakeCudaRunner(tokens=token_ids)
    scheduler_cls = MiniOverlapScheduler if args.overlap else MiniScheduler
    scheduler_kwargs = {
        "trace": args.trace,
        "max_total_tokens": args.max_total_tokens,
        "page_size": args.page_size,
        "chunked_prefill_size": args.chunked_prefill_size,
        "enable_radix_cache": args.enable_radix_cache,
    }
    if args.max_prefill_tokens > 0:
        scheduler_kwargs["max_prefill_tokens"] = args.max_prefill_tokens
    scheduler = scheduler_cls(runner, **scheduler_kwargs)
    # 阶段 3：构造请求并运行到结束。
    req = Req(
        rid="cli-0",
        origin_input_ids=[int(token_id) for token_id in input_ids],
        sampling_params=SamplingParams(
            max_new_tokens=args.max_new_tokens,
            eos_token_ids=frozenset(),
        ),
    )
    scheduler.add_request(req)
    scheduler.run_until_complete()
    # 阶段 4：教学版只输出 token ids；不加载真实 tokenizer/model。
    print(",".join(str(token) for token in req.output_ids))
    if args.trace:
        print("\n".join(runner.trace), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
