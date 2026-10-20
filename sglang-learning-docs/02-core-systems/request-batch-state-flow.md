# Req 到 ForwardBatch：请求怎样变成一轮计算

> 源码基准：上游 `16eb87c06f`；本地集成 `4f0e1cbea2`（2026-10-20）。本地日志增强与验证范围见总索引和 Mac 验证记录。

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

## 两个请求怎样跨轮存活

设 A、B 已完成 prefill，输出上限分别为 2、3。下面仅演示普通 decode：

| 轮次 | batch 中的 Req | 本轮输入 | 结果处理后 |
|---|---|---|---|
| N | A、B | 两者上轮采样 ID | 两个 output_ids 都追加 |
| N+1 | A、B 或过滤后的集合 | 尚需运行的请求 ID | A 达到停止条件，准备从后续 batch 移除 |
| N+2 | B | B 上轮 ID | B 继续，A 的可复用缓存可保留 |

“从 batch 移除”不等于立刻销毁 Req，更不等于立即把全部 KV 归零。输出消息、缓存引用和延后释放各有完成边界。

## 数据所有权与复制

```text
HTTP 请求体 ─序列化消息─> Req
                             ├─ ScheduleBatch.reqs：引用本轮请求
                             ├─ CPU 长度/停止状态
                             └─ 准备模型输入 → worker batch → ForwardBatch 张量
```

Python 列表 `list(batch.reqs)` 复制列表容器，内部仍可引用同一个 Req。只有查清调用点的 copy/filter 语义，才能判断上一轮记录是否会受到修改；不能看到变量名不同就认定完全隔离。

## 读码与验收

```bash
rg -n 'class Req|class NextBatchPlan|def filter_batch|def merge_batch' python/sglang/srt/managers/schedule_batch.py
rg -n 'def get_model_worker_batch|def forward_batch_generation' python/sglang/srt/managers
```

画出 A 结束后 B 的请求行与 KV 映射。答案应保留 B 的有效映射，移除 A 的执行参与，并区分 A 的缓存引用和请求行释放。不能把过滤操作画成“压缩后所有物理 KV 一起搬家”。

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/managers/scheduler.py](../../python/sglang/srt/managers/scheduler.py) | event_loop_normal 与 get_next_batch_to_run |
| [python/sglang/srt/managers/schedule_batch.py](../../python/sglang/srt/managers/schedule_batch.py) | Req、ScheduleBatch、filter_batch、merge_batch |
| [python/sglang/srt/model_executor/forward_batch_info.py](../../python/sglang/srt/model_executor/forward_batch_info.py) | ForwardBatch.init_new |
