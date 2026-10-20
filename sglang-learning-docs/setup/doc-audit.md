# 文档逐页审计与维护

> 源码基准：上游 `16eb87c06f`；本地集成 `4f0e1cbea2`（2026-10-20）。

原有 63 页全部纳入审计；新增本页用于维护，共 64 页。审计不是仅替换提交号：核心章节补流程和状态实例，进阶章节按当前入口解释边界，基础笔记补练习和答案。

| 文档 | 本轮处理 |
|---|---|
| [01-architecture/current-code-map.md](../01-architecture/current-code-map.md) | 重写版本比较与当前代码地图；区分本地日志增强、Host 迁移和模型专用执行器。 |
| [01-architecture/foundations.md](../01-architecture/foundations.md) | 复核启动、IPC、健康检查与源码来源；补阶段化失败定位和实操验收。 |
| [01-architecture/overview.md](../01-architecture/overview.md) | 保留请求主线，补资源约束、双请求例子及请求/存储/执行三类状态。 |
| [01-architecture/week1-detailed.md](../01-architecture/week1-detailed.md) | 按启动、进程、编码、调度和流式五次练习补交付与验收。 |
| [02-core-systems/cuda-graph-and-padding.md](../02-core-systems/cuda-graph-and-padding.md) | 核对执行分支与阶段配置；补行桶、稳定指针和图外成本。 |
| [02-core-systems/decode-batch-and-isolation.md](../02-core-systems/decode-batch-and-isolation.md) | 核对新 token/KV 时序和请求边界；补串行/并发对照的证据边界。 |
| [02-core-systems/model-execution.md](../02-core-systems/model-execution.md) | 补模型 prepare hook、shared-experts scope、Mamba COW 与模型专用 RowBucketGraphRunner。 |
| [02-core-systems/prefill-batch-and-kv.md](../02-core-systems/prefill-batch-and-kv.md) | 核对长度、预算和地址分工；补双请求 packed 输入与 position 手算。 |
| [02-core-systems/request-batch-state-flow.md](../02-core-systems/request-batch-state-flow.md) | 核对 NextBatchPlan 与普通循环；补共享引用及请求结束后的跨轮状态。 |
| [02-core-systems/runtime-context.md](../02-core-systems/runtime-context.md) | 增加 create_context/use_context、串行约束与模型私有配置归属。 |
| [02-core-systems/scheduler-and-cache.md](../02-core-systems/scheduler-and-cache.md) | 拆开策略、admission、缓存匹配和分配；补场景推演。 |
| [02-core-systems/scheduler-batch-lifecycle.md](../02-core-systems/scheduler-batch-lifecycle.md) | 补 normal/overlap、WAR 依赖、延后动作、结束/回退与暂停。 |
| [02-core-systems/scheduler-status-log-scenarios.md](../02-core-systems/scheduler-status-log-scenarios.md) | 修正日志快照字段含义；说明 cur_batch_for_debug 修复与 run_id 实测。 |
| [02-core-systems/unified-cache-and-memory.md](../02-core-systems/unified-cache-and-memory.md) | 更正 pool_host 已迁移；补 DevicePoolInfo、buffer info、混合 draft 与地址所有权。 |
| [02-core-systems/week2-detailed.md](../02-core-systems/week2-detailed.md) | 补批次、地址、缓存与多池预算实验；以真实响应和快照验收。 |
| [02-core-systems/week3-detailed.md](../02-core-systems/week3-detailed.md) | 补模型张量、采样和执行器实验；明确 MPS 与 Graph 边界。 |
| [03-advanced/disaggregation-source.md](../03-advanced/disaggregation-source.md) | 补 PD 未完成传输/释放、abort 与角色切换的生命周期。 |
| [03-advanced/hicache-source.md](../03-advanced/hicache-source.md) | 更新 Host buffer info 构造、传输消费依赖与分层命中实验。 |
| [03-advanced/mamba-source.md](../03-advanced/mamba-source.md) | 补 checkpoint/临时状态区分、逻辑到物理翻译与 forward stream COW。 |
| [03-advanced/model-gateway.md](../03-advanced/model-gateway.md) | 区分路由和引擎调度；补缓存亲和、排队、重试与 fake-worker 验证边界。 |
| [03-advanced/multi-gpu.md](../03-advanced/multi-gpu.md) | 保留矩阵分片数学，补手算 shape、多通信组和单卡对照。 |
| [03-advanced/pd-disaggregation.md](../03-advanced/pd-disaggregation.md) | 补从配置到硬件的复现顺序与异常场景验收。 |
| [03-advanced/rust-serving-and-processing.md](../03-advanced/rust-serving-and-processing.md) | 删除独立 renderer 旧架构，改为 Rust server/processor 和 typed 边界。 |
| [03-advanced/source-deep-dive.md](../03-advanced/source-deep-dive.md) | 修正 Rust 入口，补按状态和生命周期组织的专题交付。 |
| [03-advanced/speculative-and-distributed.md](../03-advanced/speculative-and-distributed.md) | 补机制组合的状态与配置验证，避免推断任意组合受支持。 |
| [03-advanced/speculative-source.md](../03-advanced/speculative-source.md) | 核对候选/接受计数；采用 DFlash V2，补混合状态与性能账本。 |
| [03-advanced/swa-source.md](../03-advanced/swa-source.md) | 补窗口边界、组件生命周期和 draft/混合状态地址约束。 |
| [04-practice/exercise-solutions.md](../04-practice/exercise-solutions.md) | 补新增题答案，保留历史 token 数来源，改用机制而非固定数值验收。 |
| [04-practice/exercises.md](../04-practice/exercises.md) | 保留 demo 简化边界；增加地址、并发、context、Host 与 Graph 调用练习。 |
| [04-practice/first-code-change.md](../04-practice/first-code-change.md) | 增加当前批次日志的真实复现、最小修复与回归例子。 |
| [04-practice/profiling.md](../04-practice/profiling.md) | 补现象到对照实验的取证路径，不虚构设备 profile。 |
| [04-practice/server-and-benchmark.md](../04-practice/server-and-benchmark.md) | 核对 CLI 入口与手动请求；补验证脚本新增 I/O 和功能范围。 |
| [04-practice/testing-and-ci.md](../04-practice/testing-and-ci.md) | 修复统一缓存测试拆分路径，明确静态/单测/实服务三层检查与平台限制。 |
| [05-reference/debugging-guide.md](../05-reference/debugging-guide.md) | 核对症状到证据链；增加当前字段兼容与本地/远端范围误判。 |
| [05-reference/faq.md](../05-reference/faq.md) | 补当前源码迁移、context 并发限制、同批证据和自动检查边界。 |
| [05-reference/glossary.md](../05-reference/glossary.md) | 补 buffer info、assembler、model-owned context、row bucket 与 checkpoint。 |
| [05-reference/math-for-llm.md](../05-reference/math-for-llm.md) | 保留矩阵/概率基础；补 shape/stride、mask 手算与可执行 softmax。 |
| [05-reference/observability-trace-and-logs.md](../05-reference/observability-trace-and-logs.md) | 标注本地增强来源、实际快照和未执行的 collector；补观察开销。 |
| [05-reference/performance-intuition.md](../05-reference/performance-intuition.md) | 核对本地 KV 配置计算；补容量/并发/TTFT/ITL 和公平对照。 |
| [05-reference/prerequisites.md](../05-reference/prerequisites.md) | 保留最小前置知识；补八单元学习路线和明确交付。 |
| [05-reference/tokenizer-internals.md](../05-reference/tokenizer-internals.md) | 保留字节/BPE/增量解码；标注历史 ID 样本并补模板/revision 复现条件。 |
| [05-reference/transformer.md](../05-reference/transformer.md) | 保留模型计算主干；补模型/运行时职责、KV 复用原理和验收题。 |
| [05-reference/triton-and-sgl-kernel.md](../05-reference/triton-and-sgl-kernel.md) | 对齐最新逻辑分组与 KDA/Cake backend；补 kernel 定位练习和平台边界。 |
| [README.md](../README.md) | 同步源码基准、学习路线、隔离环境入口和实测/未覆盖范围。 |
| [capstone/capstone.md](../capstone/capstone.md) | 补六类可复现交付、双请求/地址图、硬件边界与漂移后的更新验收。 |
| [experiences/2026-06-18-find-abstractmethod-implementation.md](../experiences/2026-06-18-find-abstractmethod-implementation.md) | 保留语言知识与原链接；补 Java 对照、可执行例子或读码练习，修正当前源码入口。 |
| [experiences/2026-06-18-python-function-signature-vs-java.md](../experiences/2026-06-18-python-function-signature-vs-java.md) | 保留语言知识与原链接；补 Java 对照、可执行例子或读码练习，修正当前源码入口。 |
| [experiences/2026-06-18-python-init-explained.md](../experiences/2026-06-18-python-init-explained.md) | 保留语言知识与原链接；补 Java 对照、可执行例子或读码练习，修正当前源码入口。 |
| [experiences/2026-06-18-python-local-import.md](../experiences/2026-06-18-python-local-import.md) | 保留语言知识与原链接；补 Java 对照、可执行例子或读码练习，修正当前源码入口。 |
| [experiences/2026-06-18-python-vs-java-object-reference.md](../experiences/2026-06-18-python-vs-java-object-reference.md) | 保留语言知识与原链接；补 Java 对照、可执行例子或读码练习，修正当前源码入口。 |
| [experiences/2026-06-18-pythonpath-explained.md](../experiences/2026-06-18-pythonpath-explained.md) | 保留语言知识与原链接；补 Java 对照、可执行例子或读码练习，修正当前源码入口。 |
| [experiences/2026-06-18-sglang-chat-request-entry.md](../experiences/2026-06-18-sglang-chat-request-entry.md) | 保留语言知识与原链接；补 Java 对照、可执行例子或读码练习，修正当前源码入口。 |
| [experiences/2026-06-18-sglang-plugins-and-entry-points.md](../experiences/2026-06-18-sglang-plugins-and-entry-points.md) | 保留语言知识与原链接；补 Java 对照、可执行例子或读码练习，修正当前源码入口。 |
| [experiences/2026-06-18-sglang-server-launch-flow.md](../experiences/2026-06-18-sglang-server-launch-flow.md) | 保留语言知识与原链接；补 Java 对照、可执行例子或读码练习，修正当前源码入口。 |
| [experiences/2026-06-19-prefill-extend-chunked-prefill-decode.md](../experiences/2026-06-19-prefill-extend-chunked-prefill-decode.md) | 保留语言知识与原链接；补 Java 对照、可执行例子或读码练习，修正当前源码入口。 |
| [experiences/2026-06-19-python-package-import-and-triton.md](../experiences/2026-06-19-python-package-import-and-triton.md) | 保留语言知识与原链接；补 Java 对照、可执行例子或读码练习，修正当前源码入口。 |
| [experiences/2026-06-19-tokenizer-vs-detokenizer-manager.md](../experiences/2026-06-19-tokenizer-vs-detokenizer-manager.md) | 保留语言知识与原链接；补 Java 对照、可执行例子或读码练习，修正当前源码入口。 |
| [experiences/2026-06-21-python-self-cls-vs-java-this-factory.md](../experiences/2026-06-21-python-self-cls-vs-java-this-factory.md) | 保留语言知识与原链接；补 Java 对照、可执行例子或读码练习，修正当前源码入口。 |
| [experiences/2026-06-21-python-truthiness-vs-java-boolean.md](../experiences/2026-06-21-python-truthiness-vs-java-boolean.md) | 保留语言知识与原链接；补 Java 对照、可执行例子或读码练习，修正当前源码入口。 |
| [experiences/README.md](../experiences/README.md) | 保留语言知识与原链接；补 Java 对照、可执行例子或读码练习，修正当前源码入口。 |
| [setup/doc-audit.md](doc-audit.md) | 记录逐页改动与来源；解释检查范围、指纹更新与复验要求。 |
| [setup/gpu-setup.md](gpu-setup.md) | 保留平台依赖核对，补 Mac 与高级 GPU 实验的硬件边界。 |
| [setup/mac-debug.md](mac-debug.md) | 重写两种环境启动、模型前置、状态日志、依赖来源和输出验收说明。 |
| [setup/mac-validation.md](mac-validation.md) | 保留历史日期证据，追加新集成与独立环境结果及测试限制。 |

## 检查工具与人工判断

| 检查 | 能证明 | 不能证明 |
|---|---|---|
| 链接/锚点与代码语法 | 目标存在、引用与语法可解析 | 命令在所有环境都能成功 |
| 关键符号契约 | 阅读入口仍存在 | 默认实例与调用语义完全不变 |
| 逐页审计 JSON | 每页有修改说明和源码依据 | 自动完成全文语义审查 |
| 源码指纹 | 相关源码/声明与审计时一致 | GPU 数值、性能和任意组合支持 |
| 单测与 MPS HTTP | 记录中的特定用例与配置通过 | 未运行的其他平台同样通过 |

`doc-audit.json` 保存各页的源码依据。`source-baseline.json` 保存固定版本和相关源码目录指纹，覆盖 SRT、kernel、Rust、Gateway、包初始化和依赖声明。

## 下一次源码更新

```text
固定新提交 → 比较实际工作区与上游 → 复核本地增强
    → 按差异更新正文/来源清单 → 运行相关测试与服务
    → 留存结果 → 最后刷新指纹与验证摘要
```

指纹失败时先审计变化，不直接改期望值。历史验证保留当时提交；新的实际结果另开日期/版本段。没有硬件就明确未验证，不用路径检查替代执行证据。
