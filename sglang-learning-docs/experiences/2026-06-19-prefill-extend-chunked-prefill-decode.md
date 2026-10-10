# Prefill、EXTEND、Chunk 与 Decode

> 源码基准：learning `2aa70e3eb4`，已合入 origin/main `6fc8d9da32`（2026-10-10）。

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

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/model_executor/forward_batch_info.py](../../python/sglang/srt/model_executor/forward_batch_info.py) | ForwardMode |
| [python/sglang/srt/managers/schedule_batch.py](../../python/sglang/srt/managers/schedule_batch.py) | prepare_for_extend/decode |
| [python/sglang/srt/managers/scheduler.py](../../python/sglang/srt/managers/scheduler.py) | chunk 请求进入/退出 |
