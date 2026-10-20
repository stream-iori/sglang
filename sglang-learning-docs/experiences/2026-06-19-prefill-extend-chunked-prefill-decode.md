# Prefill、EXTEND、Chunk 与 Decode

> 源码基准：上游 `16eb87c06f`；本地集成 `4f0e1cbea2`（2026-10-20）。本地日志增强与验证范围见总索引和 Mac 验证记录。

| 名称 | 大白话 | 本轮新增位置 |
|---|---|---|
| prefill | 为输入上下文计算状态 | 通常多个 |
| EXTEND | 在已算前缀后追加输入的执行模式 | 未命中/尚未计算的片段 |
| chunked prefill | 输入太长，分多轮 EXTEND | 每轮受 chunk 与预算限制 |
| decode | 用历史状态继续生成 | 普通路径每请求通常 1 |

```text
prompt 长 10，chunk 上限 4，无缓存：
轮 1：位置 0..3，前缀 0
轮 2：位置 4..7，前缀 4
轮 3：位置 8..9，前缀 8 → 完成 prompt
轮 4：上轮输出成为输入 → 预测下一 ID
```

本地默认 context=2048；短 prompt 不一定触发 chunk。观察 chunk 要准备合适输入并检查 chunk 配置/日志，不能把每次 generate 都描述成 chunked。

## 和 Graph 的关系

EXTEND/decode 表示“算什么”，eager/Graph 表示“怎么执行”。当前已有 prefill Graph，不能把 prefill 固定等同 eager。按 token chunk 与按层 SPLIT_PREFILL 也不同。

## 和输出的关系

生成了一个 token ID，通常还要下一轮把它作为输入，才计算该位置自己的 KV。prompt 的中间 chunk 不是正式输出一段相同数量的文本。

详见 [Prefill 与 KV](../02-core-systems/prefill-batch-and-kv.md)。

## 三个相互独立的轴

| 轴 | 例子 |
|---|---|
| 本轮算哪些输入 | EXTEND、DECODE、TARGET_VERIFY |
| 输入是否分块 | chunked prefill 的 token chunk |
| 怎样执行模型 | eager、decode/prefill Graph、特定模型分层执行 |

自测：一次 EXTEND 复用前 100 token，再处理后 10 token，叫不叫 prefill？答案：是输入上下文的继续计算；不需要没有缓存才能称 prefill。是否 Graph、是否 chunk 要看另外两轴。

SPLIT_PREFILL 是相应按层路径，不能与每轮限制 token 数的 chunk 混用。把这三个轴分清，再读 ForwardMode 分支就不容易错位。

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/model_executor/forward_batch_info.py](../../python/sglang/srt/model_executor/forward_batch_info.py) | ForwardMode |
| [python/sglang/srt/managers/schedule_batch.py](../../python/sglang/srt/managers/schedule_batch.py) | prepare_for_extend/decode |
| [python/sglang/srt/managers/scheduler.py](../../python/sglang/srt/managers/scheduler.py) | chunk 请求进入/退出 |
