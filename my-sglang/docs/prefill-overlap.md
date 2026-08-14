# 进阶：标准 SRT 的连续 prefill overlap

结论：标准 SRT 可以先 launch 当前 prefill，再处理上一个 prefill 的 CPU 结果。
`my-sglang` 当前没有实现这层并发，而是把非 decode batch 当作 barrier；这不影响
基础 generation overlap 主链的正确性。

## 两种路径

| 路径 | 标准 SRT | `my-sglang` |
|---|---|---|
| 连续 decode | launch B1，再 process B0 | 已实现 |
| 连续 prefill/EXTEND | 默认允许 launch P1，再 process P0 | 当前先 process P0，再 schedule P1 |

标准版的开关是
`SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP`，默认值为 `false`。只有显式设为
`true` 且当前、上一 batch 都是 EXTEND 时，`is_disable_overlap_for_batch()` 才在
launch 当前 batch 前先处理旧结果。

## 标准 SRT 的核心时序

```text
进入本轮：result_queue=[P0]

CPU schedule P1
CPU run/launch P1                 result_queue=[P0,P1]
CPU pop/process P0                result_queue=[P1]

下一轮 schedule P2
CPU run/launch P2                 result_queue=[P1,P2]
CPU pop/process P1                result_queue=[P2]
```

这仍然遵守基础 overlap 不变量：

```text
launch current -> FIFO process previous
```

区别是 decode 的下一轮输入依赖前一轮 sample token，需要 `FutureMap` relay；
prefill middle chunk 处理的是已经确定的 prompt suffix，重点变成 KV 与请求生命周期，
而不是生成 token relay。

## 为什么需要 `inflight_middle_chunks`

同一个长 prompt 被切成多个 chunk 时，后一个 chunk 可能已经 launch，前一个 chunk
的 CPU 结果才开始处理。此时不能把前一个 middle chunk 当作 prefill 完成：

```text
chunk C0 launched ──┐
chunk C1 launched ──┼─> request.inflight_middle_chunks > 0
process C0       ───┘   只递减计数，不输出、不结束、不释放
process last chunk       计数归零后，才提交首个生成 token 和完成状态
```

| 标准字段/标记 | 解决的问题 |
|---|---|
| `chunked_req` | 当前唯一继续切分的请求 |
| `inflight_middle_chunks` | 还有多少已 launch 的 middle chunk 未被 CPU 处理 |
| `contains_last_prefill_chunk` | 当前 batch 是否可能产生可提交的最终 prefill 结果 |
| `result_queue` | 保证各 chunk 的 CPU 结果严格 FIFO 提交 |

源码入口：

1. [`Scheduler.event_loop_overlap()`](../../python/sglang/srt/managers/scheduler.py)：先 launch 当前 batch，再处理旧 batch。
2. [`Scheduler.is_disable_overlap_for_batch()`](../../python/sglang/srt/managers/scheduler.py)：连续 EXTEND 的同步开关。
3. [`Scheduler.get_new_batch_prefill()` 所在调度流程](../../python/sglang/srt/managers/scheduler.py)：launch 前增加 `inflight_middle_chunks`。
4. [`BatchResultProcessor.process_batch_result_prefill()`](../../python/sglang/srt/managers/scheduler_components/batch_result_processor.py)：middle chunk 只递减计数，最后一段才输出和结束。

## 如果以后在教学版实现

最小改造边界不是删除一个 barrier，而是同时保证：

| 必须新增/修改 | 不变量 |
|---|---|
| prefill batch 也能进入两深度 FIFO | current launch 始终早于 previous process |
| middle-chunk 在途计数 | 旧 chunk 不能提前输出、结束或释放资源 |
| chunk 快照中的范围与 KV owner | 后续调度不能依赖尚未提交的 Python 输出状态 |
| abort/recovery 排空 | 所有在途 chunk 离队后才能释放 row/KV |

因此它应作为独立功能实现并配套故障恢复测试，不能只放开当前的非 decode barrier。
