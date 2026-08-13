# Scheduler 与 KV 概览

```text
Req -> req_pool_idx(row) -> ReqToTokenPool[row] -> KV slots/pages
                         -> FutureMap[row]      -> next decode token
```

| 状态 | 含义 |
|---|---|
| `kv_allocated_len` | 本轮已预留的 KV 逻辑长度 |
| `kv_committed_len` | launch 成功、可供后继 forward 使用的 KV 长度 |
| `output_ids` | 仅 FIFO CPU result process 后才可见的 token |

生产 Fake CUDA overlap 在 `run_batch_async()` 入队成功后立即令 `allocated == committed`；延迟的是 D2H 后 CPU 提交、finish 判断和资源释放。

```text
B0 launch -> KV committed + FutureMap producer queued
B1 launch -> 读取 B0 FutureMap token
B0 process -> output_ids / EOS / finish
last owner exits -> clear FutureMap + free row/KV
```

`pipeline_step()` 对齐 SGLang `event_loop_overlap()` 的基础 generation 分支。当前教学实现不支持 speculative、grammar 或分布式分支。
