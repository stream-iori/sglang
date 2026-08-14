# 动态流程

本页回答一个问题：为什么 B1 可以在 CPU 还没拿到 B0 的 token 时先提交？先读 [新人入门](newcomer-guide.md)；这里按时间线展开。

## 1. 同步基线：没有 overlap

```text
CPU: schedule B0 -> 调 runner -> 拿到 t0 -> output_ids += t0 -> schedule B1

下一步 B1 必须等 t0 已经回到 CPU。
```

`MiniScheduler.step()` 走这条路径。它最容易理解，但 CPU 无法在等待 B0 结果时准备 B1。

## 2. Fake CUDA overlap：两条 FIFO stream

```text
CPU                         Fake CUDA queues / execution
---                         ----------------------------
enqueue B0                  forward=[B0]       copy=[D2H B0]
enqueue B1                  forward=[B0,B1]    copy=[D2H B0,D2H B1]
wait/process B0       ->    只执行 B0: sample -> FM[row]=t0 -> D2H B0
later wait/process B1 ->    再执行 B1: gather t0 -> sample t1 -> D2H B1
```

关键点：`enqueue` 只把任务放进队列，立即返回。真正的 B1 gather 在 forward stream 执行；同一 stream 的 FIFO 保证 B0 的 `stash(t0)` 已先发生。因此它不用等 CPU 的 `output_ids += t0`。

## 3. 一个请求的逐 turn 时间线

```text
Turn 0:  CPU enqueue P0                         result_queue=[P0]

Turn 1:  CPU wait/read/process P0               result_queue=[]
         CPU enqueue D0                         result_queue=[D0]
         （P0 必须先提交，请求才变为 RUNNING）

Turn 2:  CPU enqueue D1                         result_queue=[D0,D1]
         CPU wait/read/process D0               result_queue=[D1]
         Fake stream 只推进 D0: sample t1, stash FM[row]=t1

Turn 3:  CPU enqueue D2                         result_queue=[D1,D2]
         CPU wait/read/process D1               result_queue=[D2]
         Fake stream 推进 D1: gather t1, sample t2, stash FM[row]=t2
```

这里的 `P0/D0/D1` 是 batch，不是请求。当前实现只让连续 decode 重叠：首 prefill、chunked prefill、内存不足或请求结束都会形成 barrier。稳定 decode 时 `result_queue` 最多两个 batch：新 batch 一入队，旧 batch 才被等待和处理。

## 4. 四类状态别混在一起

| 名称 | 放在哪里 | 作用 | 何时可用 |
|---|---|---|---|
| `FutureMap[row]` | 模拟设备 buffer | 把 B0 的 token 交给 B1 forward | B0 sampling 后；不等 D2H |
| `next_token_ids` | 当前 `GenerationBatchResult` | 当前 batch 的采样输出；D2H 后同字段变成 host copy | forward/sample 与 copy 后 |
| `host_tokens` | 当前 result | CPU 将要处理的 token | `copy_done` 后 |
| `Req.output_ids` | Python `Req` | 已对外确认的 token | 队首 FIFO process 后 |

`FutureMap[row]` 和 `ReqToTokenPool[row]` 只共享稳定 row 编号：前者保存 token 值，后者保存 KV slot；二者没有指针或物理内存共享。

## 5. 必须等待的边界

```text
B1 是否能 enqueue?        能：只需要把 gather 任务排在 B0 后面。
B1 的 gather 何时执行?    Fake event 依赖推进到 B1 时；FIFO 保证 B0 stash 在前。
CPU 能否读取 B0 token?    不能：必须等 copy_stream 的 copy_done。
请求能否释放 row/KV?      不能：仍可能有 in-flight batch 使用该 row。
```

结束时，最后一个 in-flight queue owner 退出后才按顺序 clear FutureMap row、释放 KV/row、调用 runner remove。

## 6. 失败时恢复什么

| 已发生的位置 | 保留还是撤回 |
|---|---|
| 已 FIFO process 并写入 `output_ids` 的 token | 保留 |
| result queue 中未 process 的 result | 丢弃 |
| 对应的 FutureMap、row、私有 KV | 清理/释放 |
| 未完成请求 | 回到 waiting，按已有 `output_ids` 重建上下文 |

这条恢复路径不模拟真实 CUDA 错误处理；它的目的只是让教学运行时的请求、KV 和 FutureMap 账目恢复一致。
