# SGLang 源码学习：以当前 SRT 为准

**整套讲义已按当前源码重写，Mac 实操统一使用标准 Torch MPS。** 学习主线先跑通 Qwen3-0.6B 的普通请求，再读缓存、地址和执行器，最后进入高级机制。

> 源码基准：learning 合并提交 `2aa70e3eb4`，包含 origin/main `6fc8d9da32`（2026-10-10）。本次文档工作没有再次 fetch；“当前”指该检出的代码，不声称已覆盖之后的远端提交。

## 当前架构与学习范围

```text
文字 / OpenAI messages
  → API / template / TokenizerManager
  → Req / Scheduler / ScheduleBatch
  → TpModelWorker / ForwardBatch / KVLocPlan
  → ModelRunner → eager / decode Graph / prefill Graph
  → 模型 / Attention / KV pools / sampling
  → DetokenizerManager → 文字 / SSE

配置：RuntimeContext        复用：UnifiedRadixCache + Full/SWA/Mamba
```

| 学习线 | 内容 | 验证范围 |
|---|---|---|
| Mac 本地 | Torch MPS、normal 调度、torch_native Attention、pytorch sampling | 本地模型生成、前缀复用、chat 与 SSE 已实测 |
| 当前源码 | 统一缓存、KVLocPlan、RuntimeContext、runner 分支、Rust 组件 | 路径/关键符号可自动核对 |
| 高级执行 | CUDA Graph、kernels、多卡、PD、HiCache、投机 | 需要匹配硬件和独立测试；本次未实测 |

## 直接开始

从仓库根目录执行：

```bash
bash sglang-learning-docs/setup/setup_mac.sh --dry-run
bash sglang-learning-docs/setup/setup_mac.sh
bash sglang-learning-docs/setup/launch_mac.sh
# 另一终端
python/.venv/bin/python sglang-learning-docs/setup/verify_mac.py
```

默认地址 `http://127.0.0.1:30000`，模型 `~/.modelscope/models/Qwen3-0.6B`。之前的 30009/30010 服务已停止，历史验证见 [mac-validation](setup/mac-validation.md)。

当前上游仍有一个 MPS 导入耦合：安装保留 mlx 核心包作兼容依赖，不安装 mlx-lm；启动强制关闭其 runner，所有本地学习与执行走标准 Torch MPS。原因与干净环境证据在 [Mac 指南](setup/mac-debug.md)。

## 推荐主线

| 顺序 | 阅读 | 必须能回答 |
|---|---|---|
| 0 | [本次源码地图](01-architecture/current-code-map.md) | 旧入口为什么不能直接套用 |
| 1 | [架构](01-architecture/overview.md) → [启动](01-architecture/foundations.md) | 一条请求经过哪些进程 |
| 2 | [请求流](02-core-systems/request-batch-state-flow.md) → [生命周期](02-core-systems/scheduler-batch-lifecycle.md) | Req、batch、plan 分别是什么 |
| 3 | [Prefill](02-core-systems/prefill-batch-and-kv.md) → [Decode](02-core-systems/decode-batch-and-isolation.md) | packed token 怎样保持边界，KV 写在哪里 |
| 4 | [统一缓存](02-core-systems/unified-cache-and-memory.md) | 树、组件、allocator、pool 的职责 |
| 5 | [执行器](02-core-systems/model-execution.md) → [Graph](02-core-systems/cuda-graph-and-padding.md) | 每轮 forward 怎样选路径，padding 为何需要 |
| 6 | [RuntimeContext](02-core-systems/runtime-context.md) → [状态日志](02-core-systems/scheduler-status-log-scenarios.md) | 怎样证明配置生效、缓存命中 |
| 7 | [高级索引](03-advanced/source-deep-dive.md) → [结业](capstone/capstone.md) | 哪些机制改变请求、地址或状态位置 |

基础问题先看 [token/UTF-8/BPE](05-reference/tokenizer-internals.md)、[Transformer](05-reference/transformer.md)、[KV 字节与性能](05-reference/performance-intuition.md)、[矩阵按输入/输出拆分](03-advanced/multi-gpu.md)。

## 全部文档索引

### 一、建立当前架构地图

| 文档 | 主题 |
|---|---|
| [current-code-map](01-architecture/current-code-map.md) | 10 月源码大改造：学习入口怎么变了 |
| [foundations](01-architecture/foundations.md) | 从启动到第一条响应 |
| [overview](01-architecture/overview.md) | 当前架构：一条请求、三种状态、两条学习线 |
| [week1-detailed](01-architecture/week1-detailed.md) | 第一阶段：启动、请求与进程 |

### 二、读懂普通请求与核心运行时

| 文档 | 主题 |
|---|---|
| [cuda-graph-and-padding](02-core-systems/cuda-graph-and-padding.md) | CUDA Graph、stream、padding 与请求边界 |
| [decode-batch-and-isolation](02-core-systems/decode-batch-and-isolation.md) | Decode 与请求隔离：拼在一起也不串上下文 |
| [model-execution](02-core-systems/model-execution.md) | ModelRunner 与 Runner：forward 现在怎样执行 |
| [prefill-batch-and-kv](02-core-systems/prefill-batch-and-kv.md) | Prefill：算哪些 token，给它们哪些 KV 地址 |
| [request-batch-state-flow](02-core-systems/request-batch-state-flow.md) | Req 到 ForwardBatch：请求怎样变成一轮计算 |
| [runtime-context](02-core-systems/runtime-context.md) | RuntimeContext：参数、状态与资源怎么分开 |
| [scheduler-and-cache](02-core-systems/scheduler-and-cache.md) | 调度与缓存：先学决策，再学复用 |
| [scheduler-batch-lifecycle](02-core-systems/scheduler-batch-lifecycle.md) | Scheduler：每轮先选，再算，再处理结果 |
| [scheduler-status-log-scenarios](02-core-systems/scheduler-status-log-scenarios.md) | 读 scheduler.status：把现象对到内存与批次 |
| [unified-cache-and-memory](02-core-systems/unified-cache-and-memory.md) | 统一缓存与内存：一棵树，多种状态 |
| [week2-detailed](02-core-systems/week2-detailed.md) | 第二阶段：批次、预算与统一缓存 |
| [week3-detailed](02-core-systems/week3-detailed.md) | 第三阶段：模型执行、采样与 Graph |

### 三、按状态变化进入高级专题

| 文档 | 主题 |
|---|---|
| [disaggregation-source](03-advanced/disaggregation-source.md) | PD 源码：先拿到容量，再接收 KV，再运行 |
| [hicache-source](03-advanced/hicache-source.md) | HiCache：命中主机状态还要恢复到设备 |
| [mamba-source](03-advanced/mamba-source.md) | Mamba/SSM：缓存的是状态，不只是历史 K/V |
| [model-gateway](03-advanced/model-gateway.md) | Model Gateway：选择 engine 的层 |
| [multi-gpu](03-advanced/multi-gpu.md) | 多卡：按输出拆、按输入拆，最后怎么合并 |
| [pd-disaggregation](03-advanced/pd-disaggregation.md) | PD 学习与验证顺序 |
| [rust-serving-and-processing](03-advanced/rust-serving-and-processing.md) | Rust 组件：服务、处理、树和网关分别是什么 |
| [source-deep-dive](03-advanced/source-deep-dive.md) | 高级源码索引：每项优化改变了哪一层 |
| [speculative-and-distributed](03-advanced/speculative-and-distributed.md) | 投机、分布式与 PD：三个不同问题 |
| [speculative-source](03-advanced/speculative-source.md) | 投机解码：候选不是已经提交的输出 |
| [swa-source](03-advanced/swa-source.md) | SWA：窗口内的 KV 和全量 KV 不同 |

### 四、运行、观察与修改

| 文档 | 主题 |
|---|---|
| [exercise-solutions](04-practice/exercise-solutions.md) | 练习答案与验证边界 |
| [exercises](04-practice/exercises.md) | 动手练习：玩具模型帮助理解，真实服务负责验证 |
| [first-code-change](04-practice/first-code-change.md) | 第一次改代码：选能完整解释与验证的小问题 |
| [profiling](04-practice/profiling.md) | Profiling：先定位哪段慢，再解释原因 |
| [server-and-benchmark](04-practice/server-and-benchmark.md) | 服务实操：先验证正确，再测性能 |
| [testing-and-ci](04-practice/testing-and-ci.md) | 测试与 CI：目录说明主题，注册说明运行环境 |

### 五、随时查阅

| 文档 | 主题 |
|---|---|
| [debugging-guide](05-reference/debugging-guide.md) | 排查：每个结论都落到一个证据 |
| [faq](05-reference/faq.md) | 当前源码常见疑问 |
| [glossary](05-reference/glossary.md) | 术语速查：把输入、状态和计算分开 |
| [math-for-llm](05-reference/math-for-llm.md) | 读推理代码需要的矩阵和概率 |
| [observability-trace-and-logs](05-reference/observability-trace-and-logs.md) | Trace、指标和状态快照：三种证据一起看 |
| [performance-intuition](05-reference/performance-intuition.md) | 性能直觉：KV 字节、算力、带宽与提交成本 |
| [prerequisites](05-reference/prerequisites.md) | 前置知识：边读边补，不必先学完所有 CUDA |
| [tokenizer-internals](05-reference/tokenizer-internals.md) | Token、字节、UTF-8 与 BPE：从文字到 embedding |
| [transformer](05-reference/transformer.md) | Transformer 的一轮推理与 SGLang 对应关系 |
| [triton-and-sgl-kernel](05-reference/triton-and-sgl-kernel.md) | 算子入口：sglang.kernels、JIT 与 AOT |

### 六、环境和验证证据

| 文档 | 主题 |
|---|---|
| [gpu-setup](setup/gpu-setup.md) | CUDA 环境：高级实验前先匹配当前依赖 |
| [mac-debug](setup/mac-debug.md) | Mac 学习环境：标准 Torch MPS |
| [mac-validation](setup/mac-validation.md) | Mac 启动验证与排查记录（2026-10-10） |

### 七、结业

| 文档 | 主题 |
|---|---|
| [capstone](capstone/capstone.md) | 结业：用一份可复现报告讲清当前 SRT |

### 读码补充

[experiences/ 索引](experiences/README.md) 包含 Python 语法、导入、启动和分词等问题笔记。原日期文件名保留，内容已重写，不沿用旧运行结论。

## 维护与复核

```bash
python3 sglang-learning-docs/setup/check_docs.py
bash -n sglang-learning-docs/setup/setup_mac.sh
bash -n sglang-learning-docs/setup/launch_mac.sh
```

`check_docs.py` 核对本地链接、代码块和核心源码符号，不能证明每个架构描述或性能结论正确。源码更新后，还需重新核对 factory、配置解析、执行分支并运行相关检查。

| 文件 | 作用 |
|---|---|
| setup/setup_mac.sh | 从上游声明安装本地依赖 |
| setup/launch_mac.sh | 固定标准 Torch MPS 启动 |
| setup/verify_mac.py | HTTP 功能验证并保存原始响应 |
| setup/check_docs.py | 文档/源码入口漂移检查 |
| setup/mac-validation.md | 实际操作、失败原因与历史验证 |

三个 `06_demo_*.py` 是玩具模型，不能代替真实 SRT 验证。所有高级特性的验证范围都在对应章节注明。
