# Mac 学习环境：标准 Torch MPS

> 源码基准：learning `2aa70e3eb4`，已合入 origin/main `6fc8d9da32`（2026-10-10）。

Apple Silicon 本地统一走标准 Torch ModelRunner，使用现有 Qwen3-0.6B。安装、启动、验证分成三步，成功安装不等于已经启动。

```text
setup_mac.sh → 依赖解析/安装/运行时检查
launch_mac.sh → normal Scheduler → Torch ModelRunner → torch_native
verify_mac.py → health / generate / cache / OpenAI chat / SSE
```

## 可直接执行

从仓库根目录运行：

```bash
bash sglang-learning-docs/setup/setup_mac.sh --dry-run
bash sglang-learning-docs/setup/setup_mac.sh
bash sglang-learning-docs/setup/launch_mac.sh
# 在另一终端
python/.venv/bin/python sglang-learning-docs/setup/verify_mac.py
```

| 项目 | 默认值 |
|---|---|
| Python | python/.venv/bin/python；不存在时 setup 创建 Python 3.13 环境 |
| 模型 | ~/.modelscope/models/Qwen3-0.6B |
| 地址 | http://127.0.0.1:30000 |
| 执行 | --device mps |
| Attention / sampling | torch_native / pytorch |
| overlap / grammar | 关闭 overlap；grammar backend none |
| token pool / context | 4096 / 2048 |

## 安装依据

setup 从当前 `python/pyproject_other.toml` 展开 runtime_common 和 srt_mps 的 Torch 配套依赖；源码通过 PYTHONPATH 使用，不修改上游 pyproject。

| 本次验证版本 | 值 |
|---|---|
| Torch / torchvision | 2.13.0 / 0.28.0 |
| torchaudio / torchcodec | 2.11.0 / 0.15.0 |
| Transformers | 5.19.0 |
| grpcio-tools / protobuf | 1.84.0 / 7.36.2 |

版本是本次实测记录；下一次安装以当前上游声明和实际解析为准。MPS runtime 要求 Torch ≥2.13.0 且 MPS 可用。

### 一个上游导入依赖

当前 Scheduler 在 MPS 平台仍无条件导入依赖 `mlx.core` 的 mixin。因此 setup 保留 `mlx` 核心包作兼容导入依赖，不安装 mlx-lm，不启用其模型执行路径；launch 强制 `SGLANG_USE_MLX=0`。

干净环境去掉这个包实测报 ModuleNotFoundError。详细堆栈和后续验证见 [操作记录](mac-validation.md)。这是当前依赖耦合，不应写成“本地同时学习两种 runner”。

## 自定义环境与端口

```bash
SGLANG_MAC_VENV=/tmp/sglang-torch-mps-env \
  bash sglang-learning-docs/setup/setup_mac.sh
SGLANG_MAC_VENV=/tmp/sglang-torch-mps-env SGLANG_MAC_PORT=30011 \
  bash sglang-learning-docs/setup/launch_mac.sh
/tmp/sglang-torch-mps-env/bin/python sglang-learning-docs/setup/verify_mac.py \
  --url http://127.0.0.1:30011 --output /tmp/sglang-mps-verification
```

可用 `SGLANG_MAC_MODEL` 指定另一个模型。换模型后需要重新验证，不能沿用 Qwen3 成功结论。

## 代理

```bash
zsh -ic 'whence -v proxy; functions proxy'
lsof -nP -iTCP:6152 -sTCP:LISTEN
HTTPS_PROXY=http://127.0.0.1:6152 HTTP_PROXY=http://127.0.0.1:6152 \
  bash sglang-learning-docs/setup/setup_mac.sh --dry-run
```

6152 是本机 proxy 函数对应端口，先查自己的定义和监听。本次代理未监听，直连完成了安装；没有修改代理配置。

## 停止和限制

前台进程用 Ctrl+C 停止，再用 `lsof -nP -iTCP:30000 -sTCP:LISTEN` 确认。30009/30010 的旧服务已按要求停止，验证记录中的地址是历史实测地址。

本地验证普通生成、前缀缓存和 HTTP 协议。CUDA Graph、GPU kernel、多卡、PD、HiCache、投机需要匹配的平台和独立实验。

## 对照源码

| 文件 | 读什么 |
|---|---|
| [sglang-learning-docs/setup/setup_mac.sh](setup_mac.sh) | 安装逻辑 |
| [sglang-learning-docs/setup/launch_mac.sh](launch_mac.sh) | 固定启动配置 |
| [sglang-learning-docs/setup/verify_mac.py](verify_mac.py) | 端点验证 |
| [python/pyproject_other.toml](../../python/pyproject_other.toml) | 上游依赖声明 |
| [python/sglang/srt/hardware_backend/mps/runtime.py](../../python/sglang/srt/hardware_backend/mps/runtime.py) | 运行时门槛 |
