# Mac 学习环境：标准 Torch MPS

> 源码基准：上游 `16eb87c06f`；本地集成 `4f0e1cbea2`（2026-10-20）。本地日志增强与验证范围见总索引和 Mac 验证记录。

**Apple Silicon 可运行真实 Qwen3-0.6B 请求，学习普通调度、KV、缓存与协议。** 本轮使用独立虚拟环境复验，不替换旧环境；结果见[当前验证](mac-validation.md#九更新后的源码与隔离环境验证)。

```text
setup_mac.sh：依赖 → 兼容检查 → MPS runtime
launch_mac.sh：normal Scheduler → Torch ModelRunner → torch_native
verify_mac.py：生成 → 缓存 → Chat/SSE → 双请求 → 后续请求 → 可选状态日志
```

## 最短启动方式

在仓库根目录执行。默认环境为 `python/.venv`，不存在时创建 Python 3.13 环境：

```bash
bash sglang-learning-docs/setup/setup_mac.sh --dry-run
bash sglang-learning-docs/setup/setup_mac.sh
bash sglang-learning-docs/setup/launch_mac.sh
```

另一个终端执行：

```bash
python/.venv/bin/python sglang-learning-docs/setup/verify_mac.py
```

模型权重需提前准备。安装脚本不下载模型，启动脚本检查 config 和 safetensors/bin 权重是否存在。

| 参数 | 默认值 |
|---|---|
| Python 环境 | python/.venv；可用 SGLANG_MAC_VENV 指定 |
| 模型 | ~/.modelscope/models/Qwen3-0.6B；可用 SGLANG_MAC_MODEL 指定 |
| 服务地址 | 127.0.0.1:30000；端口用 SGLANG_MAC_PORT 指定 |
| 执行 | device=mps、normal，关闭 overlap |
| Attention / sampling / grammar | torch_native / pytorch / none |
| token 池 / 上下文 | 4096 / 2048 |

## 直接复现本轮隔离环境

本机已存在该环境和模型；新机器先执行安装：

```bash
SGLANG_MAC_VENV=python/.venv-learning-20261020 \
  bash sglang-learning-docs/setup/setup_mac.sh
```

终端一：

```bash
SGLANG_MAC_VENV=python/.venv-learning-20261020 SGLANG_MAC_PORT=30013 \
SGLANG_LOG_SCHEDULER_STATUS_TARGET=/tmp/sglang-learning-status-20261020 \
SGLANG_LOG_SCHEDULER_STATUS_INTERVAL=0 \
  bash sglang-learning-docs/setup/launch_mac.sh --enable-metrics
```

终端二：

```bash
python/.venv-learning-20261020/bin/python sglang-learning-docs/setup/verify_mac.py \
  --url http://127.0.0.1:30013 --status-dir /tmp/sglang-learning-status-20261020 \
  --output /tmp/sglang-learning-validation-20261020
```

interval=0 用于短请求逐轮观察，会增加同步和日志开销；它是学习配置，不是性能基线。通常可以改为 0.2 秒，但极短状态可能被采样跳过。

## 依赖依据与导入边界

| 依据 | 本轮处理 |
|---|---|
| python/pyproject_other.toml | 展开 runtime_common、tracing，读取 srt_mps 的 Torch 配套依赖 |
| 标准 CUDA pyproject | 不用于 Mac 全量安装，避免安装不适用扩展 |
| MPS runtime | Torch ≥2.13.0、MPS 可用、必要的 memory API |
| mlx 核心包 | 当前 MPS Scheduler 导入依赖要求；按源码声明安装，不启用模型 runner |
| mlx-lm | 本教程标准 Torch 路径不安装 |

当前 srt_mps 已要求 `mlx>=0.32.4`，不能照搬历史环境的 0.32.3。本轮版本与完整安装快照见[依赖快照](requirements-mps-20261020.txt)；后续安装仍以当次源码声明和解析结果为准。

本轮最终环境共 165 包；tracing 的 SDK/exporter 用于运行单测，没有启动 collector。

launch 强制 `SGLANG_USE_MLX=0`。包存在只说明满足导入，不证明模型由该后端执行；用 server_info、模型类、Attention 与 cache 日志确认实际路径。

Mac 的包初始化还会安装平台 stub，因此直接按文件绕过包初始化的某些上游 CPU 测试不适用于这条正常导入链。不要为通过此类探测在 Mac 安装 CUDA Triton；具体失败证据见验证记录。

## 输出与验收

| 文件 | 作用 |
|---|---|
| environment.json | 实测 HEAD、源码工作区差异指纹、Python/依赖与模型 config 指纹 |
| server-info.json | 服务报告的配置；脚本核对四个关键有效字段 |
| generate-cold/warm.json | 原始生成与复用结果 |
| chat.json / chat-stream.json | 普通 Chat 与 SSE 原始数据 |
| isolation-serial/parallel.json | 两请求各自串行与并发对照 |
| followup.json | 完成后服务继续可用 |
| scheduler-status.json | 指定 status-dir 时，筛选本次 run_id 的原始快照 |
| summary.json | 全部断言通过后写出的摘要 |

不固定缓存必须为 197 token，也不固定所有模型都回答同一句话。当前用例要求 greedy ID 稳定、有效前缀命中、流式完整、双请求各自输出一致。

## 常见失败与停止

| 现象 | 取证与处理 |
|---|---|
| 找不到 Python 或模型 | 读启动脚本的路径提示，准备环境或指定路径 |
| health 暂时非 200 | 查加载/预热日志；脚本默认最多等 120 秒 |
| runtime 版本错误 | 对照实际 Torch 和当前 MPS 要求，不绕过检查 |
| 没有 status | 检查同一服务的目录、metrics、间隔和 run_id |
| 代理连接失败 | 先查自己的代理定义与监听端口；不照搬历史机器的端口 |

前台服务按 Ctrl+C 结束，再用 `lsof -nP -iTCP:30013 -sTCP:LISTEN` 确认。后台服务先确认 listener 属于本次启动，再向该进程发送 TERM。

CUDA Graph、CUDA kernel、多卡、PD、HiCache、SWA/SSM/投机和 Rust 前端需要对应平台的独立实验。本轮成功覆盖普通 Python HTTP + Torch MPS 主线。
