# Mac 启动验证与排查记录（2026-10-10）

**本地 Qwen3-0.6B 已在标准 Torch MPS 路径启动并通过 HTTP 推理验证。** 代码为 learning 合并提交 `2aa70e3eb4`，包含 origin/main `6fc8d9da32`。

> 状态更新：30009、30010 的历史服务已按要求停止；下文地址和响应是已完成的实测证据，不代表当前仍有服务监听。整套学习讲义现已按同一源码基准重写。

## 一、环境与结果

| 项目 | 实测 |
|---|---|
| 主机 | macOS arm64 / Apple Silicon |
| Python 环境 | 仓库的 `python/.venv`，Python 3.13.9 |
| 模型 | `/Users/stream/.modelscope/models/Qwen3-0.6B`，直接使用现有文件，未重新下载 |
| Torch / torchvision | 2.13.0 / 0.28.0 |
| torchaudio / torchcodec | 2.11.0 / 0.15.0（按上游 srt_mps 声明） |
| Transformers | 5.19.0 |
| grpcio-tools / protobuf | 1.84.0 / 7.36.2 |
| Torch MPS 地址 | `http://127.0.0.1:30009` |

| 验证项 | Torch MPS |
|---|---|
| `/health` | 200 |
| `/server_info` | 200；设备 mps |
| `/generate` | 生成 8 token，包含 `Paris` |
| 首次长 prompt 缓存命中 | 0 token |
| 相同 prompt 再次命中 | 197 token（prompt 共 198 token） |
| 两次 greedy 输出 ID | 相同 |
| OpenAI chat 非流式 | `2 + 2 = 4.` |
| OpenAI chat 流式 | 同上，收到 `[DONE]` |

197 而不是 198 与保留最后一个输入位置重新计算 logits 的前缀匹配边界相符。这里验证的是指定模型、单卡与启动配置，不代表所有模型或所有调度模式都已验证。

MPS 日志明确显示 `Qwen3ForCausalLM`、`torch_native` Attention、BF16 KV、4096-token 池、Unified Radix Cache，CUDA Graph 自动关闭。

## 二、实际操作与命令

以下均从仓库根目录执行；日志存于 `/tmp`，不属于 Git 文件，系统清理临时文件后需重新生成。

| 顺序 | 操作 | 证据／结果 |
|---|---|---|
| 1 | 阅读 setup 脚本、启动指南和上游 `pyproject_other.toml` | 旧清单与新的 srt_mps 要求不一致 |
| 2 | 检查 MPS runtime 和上游 MPS 端到端测试 | 确认版本门槛与基础启动参数 |
| 3 | 用旧环境尝试 MPS 命令 | Torch 2.9.1 被运行时检查拒绝 |
| 4 | 查询交互式 `proxy` 函数并检查监听端口 | HTTP 6152、SOCKS 6153；HTTP 端口未监听 |
| 5 | 代理尝试依赖解析，再尝试直连 | 代理连接拒绝；直连解析、下载成功 |
| 6 | 改写 setup：展开通用 runtime，并读取上游 Torch 配套版本，复用现有环境 | 不替换上游 pyproject，不装默认 CUDA 依赖 |
| 7 | 安装依赖并检查整个环境 | 发现旧 grpcio-tools 与新版 protobuf 冲突 |
| 8 | 升级 grpcio-tools，再重跑 setup | pip check 和 MPS runtime 检查通过 |
| 9 | 增加统一启动脚本，启动 MPS | 加载本地模型、完成预热 |
| 10 | 跑 MPS/runtime 与 OpenAI protocol 单测 | 45 passed，39 subtests passed |
| 11 | 验证 MPS HTTP、生成、缓存、chat 与 SSE | 全通过，原始响应保存到临时目录 |
| 12 | 同步 README、setup、FAQ、架构、调试、实践和 trace 文档 | 统一标准执行链，移除过时的“全部已验证”声明 |
| 13 | shell 语法、文档 diff 检查 | 见本次执行的检查结果 |

### 代理与安装

```bash
# 非交互 shell 没有 proxy；只读查询交互式定义
zsh -ic 'whence -v proxy; functions proxy'
lsof -nP -iTCP:6152 -sTCP:LISTEN

# 代理连接失败的尝试
HTTPS_PROXY=http://127.0.0.1:6152 HTTP_PROXY=http://127.0.0.1:6152 \
  UV_HTTP_TIMEOUT=30 uv pip install --python python/.venv/bin/python --dry-run \
  'torch==2.13.0' 'transformers==5.19.0'

# 直连 dry-run 成功，随后执行完整安装
bash sglang-learning-docs/setup/setup_mac.sh --dry-run
bash sglang-learning-docs/setup/setup_mac.sh > /tmp/sglang-mac-setup.log 2>&1

# 第一次安装后，修复原环境中遗留的工具冲突
uv pip install --python python/.venv/bin/python --upgrade grpcio-tools
# 该修复已纳入新版 setup 的 grpcio-tools>=1.84.0 要求
bash sglang-learning-docs/setup/setup_mac.sh > /tmp/sglang-mac-setup.log 2>&1
```

### 启动与验证

启动命令需要独立终端持续运行；验证命令在另一个终端执行：

```bash
SGLANG_MAC_PORT=30009 bash sglang-learning-docs/setup/launch_mac.sh \
  > /tmp/sglang-mps-upgraded.log 2>&1

```

```bash
SGLANG_USE_MLX=0 PYTHONPATH=python:test python/.venv/bin/python -m pytest \
  test/registered/mps/unit/test_mps_runtime.py \
  test/registered/unit/entrypoints/openai/test_protocol.py -q \
  > /tmp/sglang-mac-tests.log 2>&1

python/.venv/bin/python sglang-learning-docs/setup/verify_mac.py \
  --url http://127.0.0.1:30009 --output /tmp/sglang-mac-validation \
  > /tmp/sglang-mac-verification.log 2>&1

bash -n sglang-learning-docs/setup/setup_mac.sh sglang-learning-docs/setup/launch_mac.sh
git diff --check
```

## 三、排查方法与根因

| 现象 | 取证方法 | 判断与处理 |
|---|---|---|
| 启动立即报版本错误 | 读 traceback、runtime.py，查询已安装包版本 | 升级环境；不绕过版本检查 |
| `proxy` 在工具 shell 中找不到 | `zsh -ic` 查看函数来源 | 函数仅在交互式配置加载；对命令显式设置代理环境变量 |
| 代理连接拒绝 | lsof 检查端口、uv 错误链 | 本地代理未监听；直连下载安装成功 |
| pip check 报 protobuf 冲突 | 读依赖约束，升级工具后重查 | grpcio-tools 1.75.1 要求 protobuf <7；升级到 1.84.0 后相容 |
| HTTP 端口有监听但 health 为 503 | 对照时间戳、warmup 和 ready 日志 | 预热未完成；等待 ready，验证脚本最多等 120 秒 |
| `/get_server_info` 提示弃用 | 接口响应与服务器日志 | 改用 `/server_info` |
| 日志有 CUDA/Triton 不支持提示 | 查看实际 runner、device、Attention、KV 日志，再发送请求 | 可选 CUDA 后端探测提示未阻止指定 MPS 路径 |

查看日志时避免打印整行 `server_args`，只取相关事件：

```bash
rg -n 'Traceback|RuntimeError|ERROR|Load weight|KV Cache|ready to roll|cached-token' \
  /tmp/sglang-mps-upgraded.log
uv pip check --python python/.venv/bin/python
lsof -nP -iTCP:30009 -sTCP:LISTEN
```

响应目录各包含 `server-info.json`、两次 generate 响应、chat 响应、SSE chunks 和 `summary.json`。验证脚本禁用 requests 的代理继承，保证 localhost 请求不经过外部代理。

## 四、修改范围与服务管理

新增 `launch_mac.sh`、`verify_mac.py` 与本记录；改写 `setup_mac.sh`，同步指南与索引。只修改学习资料和 `python/.venv` 依赖，未修改 SRT 实现、模型权重或用户代理配置。

启动验证结束时曾保留 Torch MPS 本地服务，随后按要求停止。关闭时优先在启动终端按 Ctrl+C；如果已放在后台，先通过 lsof 查出对应 listener PID，再只向该 PID 发送 TERM，不使用匹配所有 Python 进程的命令。进程号仅对当前运行有效。

## 五、统一到 Torch MPS 的改造

安装脚本展开 `runtime_common` 和上游 `srt_mps` 中的 Torch 配套包，只验证标准 MPS runner。干净环境发现上游 `scheduler.py` 在 MPS 平台仍无条件导入依赖 `mlx.core` 的 mixin；因此保留 `mlx` 核心包作为兼容导入依赖，不安装其他模型 runner 包，不启用其他后端。启动脚本移除后端选择参数，固定标准 MPS。Trace 与 prefill/extend 讲义按标准执行链重写，保留 CUDA/TP/PD 作为远程 GPU 学习内容。

已有环境中历史安装的其他包不会由脚本自动卸载。隔离验证可使用 `SGLANG_MAC_VENV=python/.venv-mps` 安装和启动；30009 的标准服务随后已停止。

### 干净环境复验

使用 `/tmp/sglang-torch-mps-env` 从空环境安装最终脚本的依赖，未安装其他后端的模型 runner 包。首次去掉兼容核心包后报错，根因为 `scheduler.py` 的 MPS 平台导入分支无条件加载了依赖 `mlx.core` 的 mixin；保留最低导入依赖后启动成功。

```bash
SGLANG_MAC_VENV=/tmp/sglang-torch-mps-env \
  bash sglang-learning-docs/setup/setup_mac.sh
SGLANG_MAC_VENV=/tmp/sglang-torch-mps-env SGLANG_MAC_PORT=30011 \
  bash sglang-learning-docs/setup/launch_mac.sh
# 另一终端验证
/tmp/sglang-torch-mps-env/bin/python sglang-learning-docs/setup/verify_mac.py \
  --url http://127.0.0.1:30011 --output /tmp/sglang-mps-clean-validation
```

复验结果：health 200，输出包含 `Paris`，第二次请求命中 197 token，两次输出 ID 一致；chat 与流式都返回 `2 + 2 = 4.`，SSE 收到 `[DONE]`。MPS/runtime 与 protocol 单测再次全部通过（45 项）。安装检查显示 155 个包依赖相容。

本次安装排查还遇到一次 `on: command not found`：安装期间编辑了正在执行的 shell 文件，解释器恢复读取时偏移错位。固定最终脚本后重新执行，安装与验证通过；不是包或 MPS 运行时错误。

原始证据：`/tmp/sglang-torch-mps-clean-setup.log`、`/tmp/sglang-mps-clean-start.log`、`/tmp/sglang-mps-clean-validation.log`、`/tmp/sglang-mps-clean-tests.log` 和响应目录 `/tmp/sglang-mps-clean-validation`。临时 30011 服务验证后停止；原 30009 服务随后也已按要求停止。

框架名称只在必要的上游关闭开关、兼容依赖和排查根因中保留。PD 讲义现按当前源码重写，不再保留特定机器的 RDMA 网卡配置。

## 六、整套讲义重写与复核

此前只有 Mac 入口和部分段落更新，README 仍以 6 月源码为基准。本次对全部原有学习讲义重写，并新增当前代码地图、RuntimeContext、统一缓存、Graph/padding、Rust 服务链五个专题。保留本文件的实际安装/启动证据，纠正服务当前状态。

| 复核对象 | 取证手段 | 结果 |
|---|---|---|
| 当前源码基准 | git rev-parse / log | learning 2aa70e3eb4，已合入 main 6fc8d9da32；未再次 fetch |
| 普通调度 | 读取 event_loop_normal / get_next_batch_to_run | 当前返回 NextBatchPlan，取 batch_to_run |
| 本地 cache | registry 分支 + clean-start 日志 | UnifiedRadixCache、FullComponent、Python UnifiedTreeCore |
| 内存组织 | rg --files + 构造/地址源码 | allocator/ 已存在；pool/、pool_host/ 尚未建立 |
| Graph | _forward_raw + exec_.py 声明 | decode/prefill 两阶段配置，兼容性决定 Graph/eager |
| 增量解码 | detokenizer_manager.py | surr/read/sent 偏移、可打印前缀、去重 |
| 日志目标 | log_utils.py | TARGET 为目录或 stdout，非单个 JSONL 文件名 |
| 文档检查 | python3 sglang-learning-docs/setup/check_docs.py | 63 页、336 本地链接、50 源码符号，0 错误 |
| shell 与 diff | bash -n 两个脚本；git diff --check | 通过 |
| 讲义代码块 | Python AST / bash -n | 6 个 Python、29 个 bash 代码块语法通过 |
| demos | 运行三个 06_demo_*.py | scheduler finished=2，prefix 正常，ZMQ 返回 Hi! |
| 基础单测 | MPS runtime + OpenAI protocol | 45 passed、39 subtests passed；27 条依赖/弃用等警告 |
| 本地依赖 | setup --dry-run / uv pip check | 无需变更；158 包兼容 |
| KV 公式 | 读取本地模型 config.json 并计算 | BF16 112 KiB/token，4096 逻辑 token 的纯 KV 448 MiB |
| UTF-8 与 tokenizer | 增量 decoder 断言 + 本地 tokenizer 编码/解码 | 不完整字节保留；龘 分为两个 ID，单独 decode 为替换字符、整段为原字 |

重写后的主入口：[README](../README.md)。核心源码路径和符号由 check_docs.py 复核；它不验证所有语义和性能，仍需人工对照调用分支。

本次文档复核没有重新启动服务，没有执行 CUDA、多卡、PD、HiCache、投机或 Rust 服务端到端测试；普通 HTTP 成功证据来自前面的实际验证。30009/30010 已无监听进程。
