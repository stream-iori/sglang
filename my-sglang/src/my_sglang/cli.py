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
    # 默认使用本机 ModelScope 缓存中的小模型，避免每次运行都下载模型。
    return str(Path.home() / ".modelscope/models/Qwen3-0.6B")


def build_parser() -> argparse.ArgumentParser:
    # argparse 是 Python 标准库的命令行参数解析工具。
    parser = argparse.ArgumentParser(description="运行一个迷你 SGLang 风格 MLX decode。")
    parser.add_argument("--model-path", default=default_model_path())
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--overlap", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    # argv=None 时，argparse 会自动读取真实命令行参数；测试里也可以传入自定义 argv。
    args = build_parser().parse_args(argv)
    model_path = str(Path(args.model_path).expanduser())
    # tokenizer 只负责文本 <-> token id 转换；模型 forward 由 SglangMlxRunnerAdapter 处理。
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    input_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    if not input_ids:
        print("prompt produced no input tokens", file=sys.stderr)
        return 2

    # CLI 也走同一套 MiniScheduler，确保手动运行和测试覆盖的是同一条链路。
    runner = SglangMlxRunnerAdapter(model_path)
    scheduler_cls = MiniOverlapScheduler if args.overlap else MiniScheduler
    scheduler = scheduler_cls(runner, trace=args.trace)
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
    # 这里只解码新生成的 token，不把 prompt 再打印一遍。
    print(tokenizer.decode(req.output_ids, skip_special_tokens=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
