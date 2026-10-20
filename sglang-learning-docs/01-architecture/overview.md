# 当前架构：一条请求、三种状态、两条学习线

> 源码基准：上游 `16eb87c06f`；本地集成 `4f0e1cbea2`（2026-10-20）。本地日志增强与验证范围见总索引和 Mac 验证记录。

SGLang 的核心工作：把许多请求安排到设备上，复用已经算过的状态，再把生成的 token 逐步送回客户端。

```text
客户端
  │ HTTP / OpenAI JSON
  ▼
API + chat template + TokenizerManager       CPU：文字 → token ID
  │ 请求消息
  ▼
Scheduler + Req + ScheduleBatch              CPU：选请求、预算、分配槽位
  │ ForwardBatch
  ▼
TpModelWorker → ModelRunner → Runner         设备：模型、Attention、logits
  │               │
  │               └─ KV pool / SSM state pool
  ▼
采样 → token ID → DetokenizerManager → HTTP / SSE
```

图是 Python HTTP、普通文本生成的学习主线。Rust 服务、PD、投机解码会改变部分边界，见进阶章节。

## 先分清三种状态

| 状态 | 例子 | 谁管理 |
|---|---|---|
| 请求状态 | 输入 ID、输出 ID、停止条件、当前长度 | Req、TokenizerManager、Scheduler |
| 存储状态 | 哪些 KV 槽空闲、哪个前缀可复用、状态位于设备还是主机 | allocator、pool、UnifiedRadixCache |
| 执行状态 | 本轮 token、positions、序列长度、读写 KV 地址 | ForwardBatch、KVLocPlan、Runner |

请求完成，KV 可能仍留在缓存。槽位是内存位置，token ID 是词表编号，两者不能混用。

## 现在应该怎么读

| 线路 | 环境 | 目标 |
|---|---|---|
| 本地实操 | Apple Silicon + Torch MPS + Qwen3-0.6B | 跑通普通请求、前缀命中、采样、流式输出 |
| 高级读码 | 当前仓库源码；执行需要对应硬件和依赖 | CUDA Graph、TP/EP/PP、PD、HiCache、投机解码 |

本地脚本关闭 overlap，使用 torch_native Attention 和 pytorch sampling。MPS 不执行 CUDA Graph。

下一步：[基础链路](foundations.md) → [请求状态流](../02-core-systems/request-batch-state-flow.md) → [统一缓存](../02-core-systems/unified-cache-and-memory.md) → [模型执行](../02-core-systems/model-execution.md)。

## 从第一性原理看三个约束

| 约束 | 直接推导 | SRT 中的机制 |
|---|---|---|
| 同时到来的请求长度不同 | 请求不能永久绑定成一个静态 batch | continuous batching、过滤与合并 |
| 设备容量有限，历史状态很大 | 只保存值得复用、仍可能被读取的状态 | KV/SSM pool、前缀缓存、锁与驱逐 |
| 一轮计算既有固定结构又有动态数据 | 分开调度决策与执行表达 | ScheduleBatch、ForwardBatch、Graph eligibility |

Java 的“请求对象”只能帮助理解 Req 的生命周期。Python 进程之间经消息传递，不共享一个 JVM 堆；GPU tensor 又有独立的执行时序。理解引用相同，还要理解它在哪个进程、哪个 stream 上被读取。

## 一个贯穿教程的例子

```text
A 输入：共同前缀 + 问题甲    B 输入：共同前缀 + 问题乙
                 │
第 1 步：A 到达，缓存未命中 → prefill → 采样首个 ID
第 2 步：A 继续 decode；B 到达 → 重新选批次
第 3 步：B 仅在匹配且状态有效时复用前缀
第 4 步：A 结束 → 输出结束；可复用 KV 仍可能留在树里
第 5 步：B 继续 → 资源紧张时驱逐可回收缓存
```

实际轮次由策略、chunk、预算和 backend 决定。后续所有示意数值都以“说明性场景”标注，HTTP 与日志记录才是实测。

## 自测

| 问题 | 合格答案 |
|---|---|
| 为什么请求结束，空闲 KV 不必恢复原值？ | 合法前缀缓存仍占用设备槽位，需区分可驱逐与锁定 |
| 为何 Mac 成功不证明 CUDA 路径成功？ | attention、graph、通信和 kernel 走不同分支 |
| 学习到什么程度算完成？ | 用日志、响应和源码解释一个请求的状态变化，而非背文件名 |

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/managers/scheduler.py](../../python/sglang/srt/managers/scheduler.py) | 调度与初始化 |
| [python/sglang/srt/model_executor/model_runner.py](../../python/sglang/srt/model_executor/model_runner.py) | 普通执行和 Graph 路由 |
| [python/sglang/srt/mem_cache/registry.py](../../python/sglang/srt/mem_cache/registry.py) | 实际 cache 选择 |
