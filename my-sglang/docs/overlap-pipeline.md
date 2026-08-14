# Fake CUDA overlap pipeline

本文描述 `my-sglang` 当前实现：CPU 不运行 CUDA，但用确定性的双逻辑 stream 和 event 对齐 SGLang 基础 generation overlap 的可观察语义。

## 一张图

```text
row A = 3
进入稳定 decode turn：result_queue=[B0]，B0 已 enqueue，尚未 process

CPU scheduler                 forward_stream queue                  copy_stream queue
─────────────                 ────────────────────                  ─────────────────
enqueue B1                    [B0 forward, B1 forward]              [D2H B0, D2H B1]
wait B0.copy_done       --->  执行 B0 forward/sample               D2H(t0) -> host B0
                              output_tokens_buf[3] = t0             B0.copy_done ready
append A.output_ids += t0
pop B0

# B0 event 不会把 B1 提前执行；后续等 B1 时才推进
wait B1.copy_done       --->  gather output_tokens_buf[3] == t0     D2H(t1) -> host B1
                              B1 forward/sample -> t1               B1.copy_done ready
                              output_tokens_buf[3] = t1
```

这里的 overlap 是：CPU 在 B1 已提交后才等待、读取、处理 B0；不是 B0/B1
两个同一请求的 forward 同时执行。Fake CUDA 只在 event 等待时确定性地
推进必要前缀；真实 CUDA 可能自主推进已提交任务。当前实现的 B0/B1 是连续
decode batch：首 prefill 要先 FIFO process，令请求进入 `RUNNING` 后才可开始 relay。

## 四本账

| 账本 | 内容 | 谁消费 |
|---|---|---|
| FutureMap | `output_tokens_buf[row]` 与 valid bit | 下一轮 forward stream gather |
| result queue | launch snapshot + 异步 result | CPU 按 FIFO 提交 |
| Req | `output_ids`、状态、KV 水位 | scheduler 的结果处理 |
| owner 计数 | 在途 batch 对 Req 的引用数 | row/KV 安全释放 |

`row` 是请求存活期内稳定的索引，不是 batch position。`FutureMap[row]` 保存的是 token 值；它和 `ReqToTokenPool[row]` 的 KV slot 映射共享 key，但保存的资源不同。

## 实现顺序

```python
# 进入稳定 decode turn 时 result_queue == [B0]
batch = schedule_current_batch()       # B1
result = runner.run_batch_async(batch, future_map)
# forward_stream: forward -> sample -> FutureMap.stash
# copy_stream: wait forward_done -> D2H -> copy_done.record
result_queue.append((batch.copy(), result))

# 当前 B1 已 enqueue；现在才处理 B0
old_batch, old_result = result_queue[0]
tokens = runner.resolve(old_result)  # 内部 sync copy_done，再读 host buffer
process_batch_result(old_batch, tokens)
result_queue.popleft()
```

`FakeCudaEvent.synchronize()` 只执行满足该 event 的 stream 前缀；例如 resolve B0 不会把 B1 的 D2H 也提前完成。这对应 CUDA event 的依赖边界，虽然不模拟耗时。

## 生命周期边界

```text
launch 成功：KV allocated == committed；token 尚未进入 output_ids
FutureMap stash：后继可消费设备侧 token
copy_done sync：CPU 可以读取 host token
FIFO process：写 output_ids，判断 EOS/长度
请求已结束且 owner == 0：才 clear FutureMap row，并释放 row/KV
```

若 B0 令请求结束，而 B1 已提交，B1 的 token 被丢弃；请求资源必须等 B1 离开 queue 后才释放。若 resolve 失败，丢弃整条在途队列、清除 FutureMap 和物理资源，仅保留已 FIFO 提交的 token，再把未完成请求放回 waiting queue。

## 与 SGLang 的对应

| my-sglang | SGLang CUDA |
|---|---|
| `FutureMap.output_tokens_buf` | `FutureMap.output_tokens_buf` |
| `FakeCudaRunner.forward_stream` | scheduler `forward_stream` |
| `FakeCudaRunner.copy_stream` | scheduler `copy_stream` |
| `FakeCudaEvent.copy_done` | `GenerationBatchResult.copy_done` |
| `pipeline_step()` | `Scheduler.event_loop_overlap()` |

代码入口：`src/my_sglang/runner.py`、`src/my_sglang/overlap_scheduler.py`、`tests/test_overlap_scheduler.py`。
