#!/usr/bin/env bash
# 从上游 Apple Silicon 依赖声明安装；默认复用 python/.venv。
set -euo pipefail
PROJ_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJ_ROOT"
VENV_DIR="${SGLANG_MAC_VENV:-python/.venv}"
PYTHON="$VENV_DIR/bin/python"
if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
    echo '需要 macOS Apple Silicon（arm64）。' >&2
    exit 1
fi
if [[ ! -x "$PYTHON" ]]; then
    uv venv "$VENV_DIR" --python 3.13
fi
requirements=$(mktemp)
trap 'rm -f "$requirements"' EXIT
# 展开同包 extra；不安装默认 CUDA 包，也不改写上游 pyproject.toml。
"$PYTHON" - "$requirements" <<'PY'
import re
import sys
import tomllib
from pathlib import Path
project = tomllib.loads(Path('python/pyproject_other.toml').read_text())['project']
extras = project['optional-dependencies']
seen = set()
requirements = set(project['dependencies'])
def expand(name):
    if name in seen:
        return
    seen.add(name)
    for dep in extras[name]:
        if dep.startswith('sglang[') and dep.endswith(']'):
            for child in dep[7:-1].split(','):
                expand(child)
        else:
            requirements.add(dep)
# 只取标准 Torch 路径：通用 runtime 与上游 Torch 配套包。
expand('runtime_common')
for dep in extras['srt_mps']:
    if re.match(r'^[A-Za-z0-9_.-]+', dep).group() in {'torch', 'torchvision', 'torchaudio', 'torchcodec'}:
        requirements.add(dep)
# 上游 Scheduler 在 MPS 平台仍导入另一个 mixin，需保留最低导入依赖。
# 仅安装核心包，不安装该框架的模型 runner 包，不启用其执行路径。
requirements.add(next(dep for dep in extras['srt_mps'] if re.match(r'^[A-Za-z0-9_.-]+', dep).group() == 'mlx'))
# 旧学习环境的 grpcio-tools 1.75 与新版 protobuf 不兼容。
requirements.update(['pytest', 'parameterized', 'accelerate', 'socksio', 'grpcio-tools>=1.84.0'])
Path(sys.argv[1]).write_text('\n'.join(sorted(requirements)) + '\n')
PY
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
if [[ "${1:-}" == --dry-run ]]; then
    uv pip install --python "$PYTHON" --dry-run -r "$requirements"
    exit 0
elif [[ $# -gt 0 ]]; then
    echo '用法：setup_mac.sh [--dry-run]' >&2
    exit 2
fi
uv pip install --python "$PYTHON" -r "$requirements"
uv pip check --python "$PYTHON"
SGLANG_USE_MLX=0 PYTHONPATH=python "$PYTHON" - <<'PY'
import torch
import transformers
from sglang.srt.hardware_backend.mps.runtime import validate_mps_runtime
validate_mps_runtime()
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
print(f'Torch {torch.__version__}; Transformers {transformers.__version__}')
print('Torch MPS 运行时检查通过；下一步启动服务并发送请求。')
PY
