# 当前架构：一条请求、三种状态、两条学习线

> 源码基准：learning `2aa70e3eb4`，已合入 origin/main `6fc8d9da32`（2026-10-10）。

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

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/managers/scheduler.py](../../python/sglang/srt/managers/scheduler.py) | 调度与初始化 |
| [python/sglang/srt/model_executor/model_runner.py](../../python/sglang/srt/model_executor/model_runner.py) | 普通执行和 Graph 路由 |
| [python/sglang/srt/mem_cache/registry.py](../../python/sglang/srt/mem_cache/registry.py) | 实际 cache 选择 |
