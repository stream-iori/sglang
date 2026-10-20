# Mac 启动验证与排查记录（最近复验：2026-10-20）

> 当前教程基准：上游 `16eb87c06f`，本地集成 `4f0e1cbea2`。新基准的安装、HTTP、状态日志和测试证据见[第九节](#九更新后的源码与隔离环境验证)。

**以下第一至八节是历史操作记录。** 各段的提交、依赖、目录状态和“尚未对齐”描述只对应当时阶段；不能用于判断更新后源码。历史服务均已停止。

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

## 七、2026-10-20 本地复验

**当前本地代码可直接运行标准 Torch MPS 学习流程。** 本次复用已有虚拟环境与模型，没有重新安装依赖；验证后已停止临时服务，30012 无监听。

| 对象 | 证据与结果 |
|---|---|
| 源码对齐 | HEAD `d982b33439`；相对 `2aa70e3eb4`，`python/sglang/srt`、`python/sglang/kernels`、`python/pyproject.toml`、`python/pyproject_other.toml` 无差异 |
| 对齐边界 | 未 fetch；只确认本地检出与讲义基准一致，未确认远端最新 main |
| 主机与依赖 | macOS 27.0.1 / arm64，Python 3.13.9，Torch 2.13.0，Transformers 5.19.0；MPS 可用，158 个包依赖相容 |
| 模型与执行 | 本地 Qwen3-0.6B；server_info 确认 mps / torch_native / pytorch，关闭 overlap |
| 实际缓存 | 日志确认 UnifiedRadixCache、FullComponent、UnifiedTreeCore；BF16 KV，4096-token 池 |
| HTTP 与生成 | health 200；生成 8 token，文本包含 Paris |
| 前缀复用 | 首次 cached_tokens=0，再次为 197；两次 greedy 输出 ID 一致 |
| OpenAI Chat / SSE | 均返回 `2 + 2 = 4.`；流式收到 `[DONE]` |
| 基础单测 | MPS runtime + OpenAI protocol：45 passed，39 subtests passed，27 warnings |
| 教学 demo | 三个 demo 全部运行完成：scheduler finished=2，前缀匹配正常，ZMQ 返回 Hi! |
| 文档与 shell | 63 页、336 本地链接、50 源码符号，0 错误；两个 shell 脚本语法检查通过 |

复现命令（启动与 HTTP 验证在不同终端执行）：

```bash
SGLANG_MAC_PORT=30012 bash sglang-learning-docs/setup/launch_mac.sh \
  > /tmp/sglang-mps-audit-20261020.log 2>&1
```

```bash
python/.venv/bin/python sglang-learning-docs/setup/verify_mac.py \
  --url http://127.0.0.1:30012 --output /tmp/sglang-mps-audit-20261020

SGLANG_USE_MLX=0 PYTHONPATH=python:test python/.venv/bin/python -m pytest \
  test/registered/mps/unit/test_mps_runtime.py \
  test/registered/unit/entrypoints/openai/test_protocol.py -q \
  > /tmp/sglang-mps-audit-tests-20261020.log 2>&1
```

原始响应在 `/tmp/sglang-mps-audit-20261020/`，包括 server-info、cold/warm、chat、SSE 和 summary JSON。启动日志与单测日志路径见上面的命令；临时文件可能被系统清理，本节保留关键结果。

本次没有验证 CUDA Graph、CUDA kernel、多卡、PD、HiCache、投机或 Rust 服务端到端行为。普通 MPS 请求成功不能替代这些机制的硬件验证。

## 八、2026-10-20 远端差异复核

**本地源码未变，不代表远端 SRT 未变；本讲义尚未对齐远端最新 main。** 第七节的源码一致与 MPS 成功结论仅适用于本地 HEAD。

| 证据 | 结果与范围 |
|---|---|
| `git ls-remote origin refs/heads/main` | 查询时 main 为 `16eb87c06fca4b14f91ddc1503e2353f70094ff2` |
| GitHub compare API | 相对 `6fc8d9da32`，ahead_by=677、behind_by=0；这是全仓库提交数，不是 SRT 提交数 |
| fetch 后的实际源码比较 | 已执行 `git fetch --no-tags origin main`，仅更新远端引用，未切换或合并工作区；`git diff --shortstat 6fc8d9da32 origin/main -- python/sglang/srt` 为 687 个文件变化，新增 52687 行、删除 16233 行 |
| Host KV 池实际 diff | `4ec19ca27f` / PR #40917 修改 35 个文件；新增 buffer info 描述，Host 池构造由 device_pool 改为 buffer_info 等显式输入，已存在 `mem_cache/pool_host/` |
| 其他提交线索 | 已有统一内存支持不对称 KV 行、draft 与 SWA/循环状态层组合、prefill Graph 等变化，仍需逐项核对当前完整源码 |
| 验证边界 | 没有切换工作区到远端版本，也没有对远端版本执行 Mac 启动验证；不能沿用第七节的成功结果 |

来源：[基准到查询时 main 的比较](https://github.com/sgl-project/sglang/compare/6fc8d9da3288e9710a6f9a1f59503cacf8a984f0...16eb87c06fca4b14f91ddc1503e2353f70094ff2)、[Host KV 池改造](https://github.com/sgl-project/sglang/pull/40917)、[不对称 KV 行](https://github.com/sgl-project/sglang/pull/43325)、[draft 与混合状态层](https://github.com/sgl-project/sglang/pull/43326)。

compare API 的文件列表最多返回 300 项，本次返回恰好 300 项，不能据此统计全部 SRT 改动；上述 Host 池结论另外读取了单提交文件 diff。后续若以最新 main 学习，需要重新审计源码入口、更新相关正文，并独立验证运行环境。

## 九、更新后的源码与隔离环境验证

### 代码与工作区保护

| 项目 | 记录 |
|---|---|
| 固定上游 | `16eb87c06fca4b14f91ddc1503e2353f70094ff2` |
| 合并提交 | `4f0e1cbea2ce03146bc3f591435d1ddbd4d44483` |
| 实测时 HEAD | `e30718d5fc1f624994f806a77df8924ad3b06a4f`；合并后的两个额外提交仅涉及 my-smg 文档，SRT 与合并提交一致，另含当时以工作区补丁运行的日志字段修复（现已提交 `9322b92a47`） |
| 合并前保护 | 备份分支 `backup/learning-pre-srt-20261020`；保留 stash `22cc268002814a750868a95eb7679abf0aa4d406` |
| 工作区恢复 | 合并前备份 141 个路径，恢复后按 SHA256/删除状态逐项核对，0 差异；后续其他文档提交独立保留 |
| 本地增强 | struct_log、trace helpers、scheduler.status 详细摘要和 metrics 调用保留 |
| 合并冲突 | 3 个历史 kernel 位置冲突；核对无外部调用后清理旧文件，采用上游入口；另清理历史 hicache、MoE、DFlash 残留 |
| 日志修复 | cur_batch 读取改为 cur_batch_for_debug，JSON 字段不变；旧实现回归失败，新实现通过 |

备份补充文件位于 `/tmp/sglang-learning-backup-20261020-mfdx9inq/`，包含工作区/索引 binary patch、文件归档与哈希清单。保留备份分支与 stash，未推送远端。

### 当前 MPS 功能结果

| 检查 | 实测结果 |
|---|---|
| 独立环境 | `python/.venv-learning-20261020`；保留旧 `python/.venv` |
| Python / 平台 | Python 3.13.9，macOS 27.0.1，arm64 |
| 核心版本 | Torch 2.13.0，Transformers 5.19.0，mlx 0.32.4；标准 Torch 执行，未安装 mlx-lm |
| 最终依赖 | 初始 155 包；加入源码声明中的 tracing 依赖后 165 包兼容，opentelemetry-sdk 1.45.1；随后重启服务复验 |
| 模型 | 本地 Qwen3-0.6B，复用原权重；config SHA256 见保存的 environment |
| 地址 | 临时 `http://127.0.0.1:30013`；最终复验后已停止，确认无监听 |
| 有效配置 | mps / torch_native / pytorch / disable_overlap_schedule=true |
| 设备状态 | Qwen3ForCausalLM、BF16 KV、4096-token 池；UnifiedRadixCache、Full 组件、UnifiedTreeCore |
| health / generate | HTTP 200；8 个输出 token，文本包含 Paris |
| 前缀复用 | 最终重启复验 cold=0、warm=212；首次新基准复验 warm=210；两次各自 greedy ID 相同，唯一 run_id 改变 token 数，不要求历史的 197 |
| Chat / SSE | 均返回 `2 + 2 = 4.`，SSE 收到 DONE |
| 双请求 | A/B 并发输出 ID 分别与串行参考一致 |
| 生命周期 | 两个请求后继续生成，新请求成功 |
| 状态日志 | 最终 run_id `mac-audit-d1f528395807`；49 条快照，包含 EXTEND、DECODE，观察到双请求同批；cur_batch、请求行、cache 字段通过 |
| 教学 demo | scheduler finished=2，前缀匹配正常，ZMQ 返回 Hi! |
| 手动 Chat 示例 | model=default 的流式请求 HTTP 200，并收到 DONE |

状态日志 interval=0，且启用 metrics。这是功能和状态学习配置，没有做性能 benchmark。

### 测试范围与已知平台限制

初次运行覆盖 MPS runtime、OpenAI protocol、状态日志、req_time_stats、trace、RuntimeContext/override、kernel namespace。旧环境与新隔离环境都出现同一个上游探测用例失败：

`test_amx_backend_probe_is_lazy[srt/utils/common.py]` 通过 find_spec 取得文件路径后直接执行文件，绕过 `sglang/__init__.py`。该文件顶层导入 Triton；本机没有 CUDA Triton，而正常 MPS 导入会先安装平台 stub。因此失败发生在 AMX 惰性探测断言之前，是该原始文件加载方式的 Mac 限制。

相关源码和测试相对固定上游均无修改。没有通过安装 CUDA 包、修改生产分支或删除断言使它假通过。后续适用测试只排除这一个 node；完整失败日志保留。最初 trace 用例因缺少可选 SDK 跳过，随后 setup 加入上游 tracing 声明中的依赖，另行复验，结果见本节末尾。

复现适用测试：

```bash
SGLANG_USE_MLX=0 PYTHONPATH=python:test python/.venv-learning-20261020/bin/python -m pytest -q   test/registered/mps/unit/test_mps_runtime.py   test/registered/unit/entrypoints/openai/test_protocol.py   test/registered/unit/utils/test_scheduler_status_logger.py   test/registered/unit/observability/test_req_time_stats.py   test/registered/unit/observability/test_trace.py   test/registered/unit/test_runtime_context.py   test/registered/unit/test_runtime_context_override.py   test/registered/unit/kernels/test_kernels_namespace.py   --deselect='registered/unit/kernels/test_kernels_namespace.py::test_amx_backend_probe_is_lazy[srt/utils/common.py]'
```

node ID 相对于 pytest 的 test 根目录，以 `registered/` 开头；误用 `test/registered/` 不会排除该 node。已用 collect-only 确认只排除 1 项，另一项同名探测仍执行。

### 复现入口与证据保存

安装、启动、状态开关和 HTTP 验证命令见 [Mac 指南](mac-debug.md#直接复现本轮隔离环境)。完整依赖版本见[环境快照](requirements-mps-20261020.txt)，持久摘要见[本轮结果 JSON](validation-20261020.json)。

| 原始证据 | 位置 |
|---|---|
| 合并/恢复 | /tmp/sglang-learning-merge-20261020.log、sglang-learning-restore-20261020.log |
| 干净安装/trace SDK | /tmp/sglang-learning-setup-20261020.log、sglang-learning-setup-tracing-20261020.log |
| 服务日志 | /tmp/sglang-learning-server-20261020.log；最终 /tmp/sglang-learning-final-server-20261020.log |
| HTTP 与状态响应 | 首次 /tmp/sglang-learning-validation-20261020/；最终 /tmp/sglang-learning-final-validation-20261020/ |
| 原始快照 | 首次 /tmp/sglang-learning-status-20261020/；最终 /tmp/sglang-learning-final-status-20261020/ |
| 初始失败 | /tmp/sglang-learning-tests-20261020.log、sglang-learning-clean-tests-20261020.log |
| 适用测试 | SDK 安装前 /tmp/sglang-learning-applicable-tests-20261020.log；最终 /tmp/sglang-learning-final-tests-20261020.log |
| tracing 单独复验 | /tmp/sglang-learning-tracing-tests-20261020.log |
| 字段修复前/后 | /tmp/sglang-learning-status-before-20261020.log、sglang-learning-status-after-20261020.log |

没有执行 CUDA kernel/Graph、多卡、PD、HiCache、SWA/SSM/投机、Rust 前端或 OTLP collector 端到端实验。HTTP/单测通过不扩大到这些机制。

### 最终验收

| 检查 | 结果 |
|---|---|
| 最终环境全部适用测试 | **384 passed，201 subtests passed，1 deselected，32 warnings**；没有 skip |
| 单独 tracing 复验 | 安装 SDK 后 69 passed、11 subtests passed；仍未运行 collector |
| 文档结构/来源 | 64 页、408 本地链接/锚点、58 源码符号、53 代码块，0 错误 |
| 检查器负向样例 | 临时副本中的坏锚点、Python 语法错误、源码指纹不符、未审计新页均被拒绝 |
| Python 教学片段 | 11 段独立示例执行通过；self/调度器主干摘录只做语法检查 |
| 启动错误路径 | 缺 Python、缺 config、缺权重均明确失败；非法 setup 参数返回用法错误 |
| 依赖与 shell | 最终 setup dry-run 无需变更，165 包兼容；shell 语法、diff 检查通过 |
| 服务清理 | 两次临时服务均正常 TERM 退出，30013 无监听 |

一次启动错误检查还发现 macOS Bash 在未加花括号的变量后紧接中文标点时误解析变量名；提示字符串改用 `${PYTHON}`、`${model}`，错误路径复验通过。正常推理路径不受该提示修复影响。

当前讲义与固定上游加本地增强对齐，Mac 验证对应最终安装环境和保存的源码差异指纹。此结论不声称覆盖之后的新 main，也不替代高级硬件验证。
