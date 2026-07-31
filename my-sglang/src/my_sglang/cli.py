"""运行单个请求的命令行学习入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from transformers import AutoTokenizer

from my_sglang.models import Req, SamplingParams
from my_sglang.overlap_scheduler import MiniOverlapScheduler
from my_sglang.runner import SglangMlxRunnerAdapter
from my_sglang.scheduler import MiniScheduler


def default_model_path() -> str:
    """返回示例默认使用的本地小模型目录。"""
    # 默认使用本机 ModelScope 缓存中的小模型，避免每次运行都下载模型。
    return str(Path.home() / ".modelscope/models/Qwen3-0.6B")


def build_parser() -> argparse.ArgumentParser:
    """声明 CLI 参数；独立函数便于测试解析行为。"""
    parser = argparse.ArgumentParser(description="运行一个迷你 SGLang 风格 MLX decode。")
    parser.add_argument("--model-path", default=default_model_path())
    parser.add_argument("--prompt", required=True)
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
    model_path = str(Path(args.model_path).expanduser())
    # 阶段 1：tokenizer 只做文本与 token id 转换。
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    input_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    if not input_ids:
        print("prompt produced no input tokens", file=sys.stderr)
        return 2
    # 阶段 2：创建模型 runner 和普通/overlap scheduler。
    runner = SglangMlxRunnerAdapter(
        model_path, disable_radix_cache=not args.enable_radix_cache
    )
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
            eos_token_ids=frozenset(
                [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
            ),
        ),
    )
    scheduler.add_request(req)
    scheduler.run_until_complete()
    # 阶段 4：只解码新生成 token，不重复打印 prompt。
    print(tokenizer.decode(req.output_ids, skip_special_tokens=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
