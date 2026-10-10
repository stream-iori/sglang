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
exec "$PYTHON" -m sglang.launch_server \
    --model-path "$model" --host 127.0.0.1 --port "${SGLANG_MAC_PORT:-30000}" \
    --device mps --disable-overlap-schedule --grammar-backend none \
    --attention-backend torch_native --sampling-backend pytorch \
    --mem-fraction-static 0.6 --max-total-tokens 4096 --context-length 2048 "$@"
