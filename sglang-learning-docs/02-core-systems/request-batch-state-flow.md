# Req 到 ForwardBatch：请求怎样变成一轮计算

> 源码基准：learning `2aa70e3eb4`，已合入 origin/main `6fc8d9da32`（2026-10-10）。

同一个请求会参加多轮计算；每轮批次只是当时的一组请求和执行数据。

```text
新 Req ─→ waiting_queue ─→ EXTEND 批次 ─→ 本轮结果
                              │              │
                    prompt 未算完         完成 prefill
                              ↓              ↓
                         下轮 chunk      running_batch
                                             ↓
                                       DECODE → 结果
                                             │
                            ┌────────────────┴───────────────┐
                         未完成                           完成/中止
                            ↓                                ↓
                       下轮 DECODE                    输出、缓存、释放
```

| 对象 | 持有什么 | 不应误认为 |
|---|---|---|
| Req | 一个请求的输入、输出、停止和 KV 生命周期信息 | 一轮 GPU 计算 |
| ScheduleBatch | 当前参与调度的 Req 集合及执行准备信息 | 永久绑定的一群请求 |
| ForwardBatch | 本轮模型用的张量、模式、地址和 Attention 元数据 | HTTP 请求体 |
| NextBatchPlan | batch_to_run、running_batch 等调度结果 | get_next_batch_to_run 直接返回 batch |

当前 `event_loop_normal` 的关键代码可概括为：

```python
plan = self.get_next_batch_to_run(
    running_batch=self.running_batch, last_batch=self.last_batch)
self.running_batch = plan.running_batch
batch = plan.batch_to_run
if batch:
    result = self.run_batch(batch)
    self.process_batch_result(batch, result)
self.last_batch = batch
```

这是主干摘录；实际代码还包含接收、暂停、idle 和一致性检查。

`last_batch` 是上一轮执行记录，`running_batch` 是持续 decode 的集合，二者可能共享 Req 引用。chunk 尚未完成的请求不会被当作正常已完成 prefill 的请求直接合入 decode。

依次读：[批次生命周期](scheduler-batch-lifecycle.md) → [Prefill 与地址](prefill-batch-and-kv.md) → [Decode 与隔离](decode-batch-and-isolation.md)。

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/managers/scheduler.py](../../python/sglang/srt/managers/scheduler.py) | event_loop_normal 与 get_next_batch_to_run |
| [python/sglang/srt/managers/schedule_batch.py](../../python/sglang/srt/managers/schedule_batch.py) | Req、ScheduleBatch、filter_batch、merge_batch |
| [python/sglang/srt/model_executor/forward_batch_info.py](../../python/sglang/srt/model_executor/forward_batch_info.py) | ForwardBatch.init_new |
