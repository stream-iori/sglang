# Scheduler 与 KV 概览

```text
Req -> req_pool_idx(row) -> ReqToTokenPool[row] -> KV slots/pages
                         -> FutureMap[row]      -> next decode token
```

| 状态 | 含义 |
|---|---|
| `req.kv.kv_allocated_len` | 本轮已预留的 KV 逻辑长度 |
| `kv_committed_len` | launch 成功、可供后继 forward 使用的 KV 长度 |
| `output_ids` | 仅 FIFO CPU result process 后才可见的 token |

生产 Fake CUDA overlap 在 `run_batch_async()` 入队成功后立即令 `allocated == committed`；延迟的是 D2H 后 CPU 提交、finish 判断和资源释放。

```text
B0 launch -> KV committed + FutureMap producer queued
B1 launch -> 把 FutureMap gather 排在 B0 producer 后
B1 forward execution -> 读取 B0 FutureMap token
B0 process -> output_ids / EOS / finish
finished request's last owner exits -> clear FutureMap + free row/KV
```

`pipeline_step()` 对齐 SGLang `event_loop_overlap()` 的基础 generation 分支。
标准版的连续 prefill 流水线见 [prefill overlap](prefill-overlap.md)。
