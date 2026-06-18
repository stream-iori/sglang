#!/bin/bash
# SGLang Mac ARM64 学习环境一键搭建脚本
# 前提: 已安装 uv (https://docs.astral.sh/uv/)
# 用法: cd sglang && bash sglang-learning-docs/setup/setup_mac.sh
# 如需代理: HTTPS_PROXY=http://127.0.0.1:7890 bash sglang-learning-docs/setup/setup_mac.sh

set -e

PROJ_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJ_ROOT"

VENV_DIR="python/.venv"
PYTHON="$VENV_DIR/bin/python"

echo "=== 1/4 创建 venv (在 python/.venv，方便 IDE 索引) ==="
uv venv "$VENV_DIR" --python 3.13
echo "  -> $VENV_DIR"

echo "=== 2/4 安装 CPU PyTorch ==="
UV_HTTP_TIMEOUT=600 uv pip install --python "$PYTHON" \
    torch --index-url https://download.pytorch.org/whl/cpu

echo "=== 3/4 安装 SGLang 核心依赖 (不含 CUDA) ==="
UV_HTTP_TIMEOUT=600 uv pip install --python "$PYTHON" \
    numpy pydantic fastapi pyzmq aiohttp requests pillow \
    "transformers==5.8.1" accelerate \
    pybase64 orjson msgspec interegular partial_json_parser "outlines==0.1.11" \
    IPython setproctitle packaging einops scipy tiktoken sentencepiece \
    prometheus-client psutil "openai>=1.0" torchvision \
    compressed-tensors gguf dill \
    mlx mlx-lm \
    datasets uvicorn watchfiles uvloop soundfile python-multipart \
    pytest parameterized

echo "=== 4/4 验证 ==="
PYTHONPATH="python" "$PYTHON" -c "
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey, TreeNode
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.server_args import ServerArgs
print()
print('All core modules imported successfully!')
" 2>&1 | grep -v "Warning\|warning"

echo ""
echo "=== 环境搭建完成! ==="
echo ""
echo "激活环境:  source python/.venv/bin/activate"
echo ""
echo "=== 测试命令 ==="
echo ""
echo "# 跑单个测试文件:"
echo "PYTHONPATH=\"python\" python -m pytest test/registered/unit/entrypoints/openai/test_protocol.py -v"
echo ""
echo "# 跑全量 Mac 可用单元测试 (~2756 passed, ~11 分钟):"
echo "PYTHONPATH=\"python:test\" python -m pytest test/registered/unit/ --tb=short -q -k 'not test_memory_allocated' --ignore=test/registered/unit/mem_cache/test_hicache_nixl_storage.py --ignore=test/registered/unit/spec/test_ngram_corpus.py --ignore=test/registered/unit/batch_invariant_ops/"
