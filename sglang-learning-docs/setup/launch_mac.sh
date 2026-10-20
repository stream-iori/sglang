#!/usr/bin/env bash
# 用法：launch_mac.sh [额外 server 参数]；固定使用标准 Torch MPS。
set -euo pipefail
PROJ_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJ_ROOT"
PYTHON="${SGLANG_MAC_VENV:-python/.venv}/bin/python"
export PYTHONPATH="$PROJ_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
# 避免继承其他终端的后端开关；此变量名由上游定义。
export SGLANG_USE_MLX=0
model="${SGLANG_MAC_MODEL:-$HOME/.modelscope/models/Qwen3-0.6B}"
if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
    echo '需要 macOS Apple Silicon（arm64）。' >&2
    exit 1
fi
if [[ ! -x "$PYTHON" ]]; then
    echo "Python 环境不存在：${PYTHON}；请先执行 setup_mac.sh。" >&2
    exit 1
fi
if [[ ! -f "$model/config.json" ]]; then
    echo "模型目录缺少 config.json：${model}；请准备权重或设置 SGLANG_MAC_MODEL。" >&2
    exit 1
fi
"$PYTHON" - "$model" <<'PY'
import sys
from pathlib import Path
from sglang.srt.hardware_backend.mps.runtime import validate_mps_runtime
validate_mps_runtime()
model = Path(sys.argv[1])
if not any(model.glob('*.safetensors')) and not any(model.glob('pytorch_model*.bin')):
    raise SystemExit(f'模型目录缺少权重文件：{model}')
PY
exec "$PYTHON" -m sglang.launch_server \
    --model-path "$model" --host 127.0.0.1 --port "${SGLANG_MAC_PORT:-30000}" \
    --device mps --disable-overlap-schedule --grammar-backend none \
    --attention-backend torch_native --sampling-backend pytorch \
    --mem-fraction-static 0.6 --max-total-tokens 4096 --context-length 2048 "$@"
