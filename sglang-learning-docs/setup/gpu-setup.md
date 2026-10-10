# CUDA 环境：高级实验前先匹配当前依赖

> 源码基准：learning `2aa70e3eb4`，已合入 origin/main `6fc8d9da32`（2026-10-10）。

Mac 学习不需要安装 CUDA。只有准备执行 CUDA Graph、CUDA kernel 或多卡实验时，才搭建匹配的 Linux GPU 环境。

| 先核对 | 证据 |
|---|---|
| GPU 与驱动 | nvidia-smi |
| Torch CUDA 版本 | torch.__version__ / torch.version.cuda |
| 上游依赖 | 当前 python/pyproject.toml 和安装文档 |
| 模型与显存预算 | 权重、KV、激活、Graph/workspace |
| backend 能力 | 模型、dtype、硬件和当前参数检查 |

```bash
nvidia-smi
python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda)
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```

本讲义不保留旧版 CUDA 12.1/Torch 安装组合。版本组合按当前检出的依赖约束解析，具体 wheel 和驱动要求需在目标机器确认。

## 实验逐步扩大

```text
单 GPU 普通生成
    → 相同负载 eager/Graph 对照
    → 一种并行方式（如 TP）
    → 特定高级机制
    → 支持的组合
```

每一步保存命令、依赖版本、模型 revision、server_info、原始响应、日志和 benchmark 参数。没有目标 GPU 的实验，在本文仅作为待执行方案。

Mac 使用 [MPS 安装脚本](mac-debug.md)，不要拿该脚本去安装 Linux CUDA 环境。

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/pyproject.toml](../../python/pyproject.toml) | 当前标准依赖 |
| [python/sglang/srt/model_executor/model_runner_components/attention_backend_setup.py](../../python/sglang/srt/model_executor/model_runner_components/attention_backend_setup.py) | backend 选择 |
| [python/sglang/srt/model_executor/model_runner_components/cuda_graph_setup.py](../../python/sglang/srt/model_executor/model_runner_components/cuda_graph_setup.py) | Graph 支持条件 |
