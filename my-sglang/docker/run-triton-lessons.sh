#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
import torch
import triton
import os

print(f"torch={torch.__version__} triton={triton.__version__}")
print(f"TRITON_INTERPRET={os.environ.get('TRITON_INTERPRET')}")
if os.environ.get("TRITON_INTERPRET") != "1":
    raise SystemExit("ERROR: this Docker workflow requires TRITON_INTERPRET=1")
PY

# Small irregular dimensions keep interpreter execution fast while exercising every mask.
python examples/triton/01_vector_add.py --n 19 --block-size 16 --num-warps 1
python examples/triton/02_fused_elementwise.py --n 19 --block-size 16 --num-warps 1
python examples/triton/03_row_softmax.py --rows 3 --cols 7 --num-warps 1
python examples/triton/04_rmsnorm.py --rows 3 --cols 7 --num-warps 1
python examples/triton/05_matmul.py --m 5 --n 7 --k 9 --block-m 16 --block-n 16 --block-k 16 --num-warps 1
python examples/triton/06_autotune_matmul.py --m 5 --n 7 --k 9

python examples/triton/07_attention.py \
  --batch 1 --heads 2 --context 9 --head-dim 8 --block-n 8 --num-warps 1

echo "PASS all Triton CPU-interpreter lessons"
