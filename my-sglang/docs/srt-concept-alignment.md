# 与标准 SGLang SRT 的概念对照

结论：`my-sglang` 对齐基础 generation 路径的职责、因果、所有权，以及有直接
对应关系的核心字段名和对象层级；它不是可替换标准 SRT 的 API，也不复制与教学
主链无关的全部 serving 字段。

```text
标准 overlap 主链

schedule current B1
        │
        ▼
run/enqueue B1 ──────> FutureMap.stash(token) ──> next forward gather
        │
        ▼
result_queue.append(B1)
        │
        ▼
pop/process previous B0 ─> copy_done.synchronize() ─> output_ids / finish
```

| 概念 | `my-sglang` | 标准 SRT 证据 | 结论 / 边界 |
|---|---|---|---|
| overlap 顺序 | enqueue current，再 FIFO process previous | [`Scheduler.event_loop_overlap()`](../../python/sglang/srt/managers/scheduler.py) | 基础 generation 因果对齐 |
| token relay | `FutureMap.stash/gather` | [`FutureMap.stash()` / `resolve_forward_inputs()`](../../python/sglang/srt/managers/overlap_utils.py) | 按稳定 `req_pool_idx` relay token 对齐 |
| `publish` 含义 | 教学版无独立 `publish()` | [`FutureMap.publish()`](../../python/sglang/srt/managers/overlap_utils.py) | 标准版 `publish()` 主要写 seq_lens/event；token 是 `stash()`，不得混用 |
| FutureMap 安全检查 | bool valid，gather 后失效 | `SGLANG_IS_IN_CI` 下 `_assert_nonneg_and_invalidate()` | 对齐 CI consume-once 目的；不是标准生产数据结构 |
| D2H 边界 | fake copy stream + `copy_done` | [`GenerationBatchResult.copy_to_cpu()`](../../python/sglang/srt/managers/utils.py) | 因果对齐；Fake event 只在 sync 时推进，不模拟真实并发 |
| result queue | `_PipelineJob` FIFO，瞬时深度不超过 2 | `batch.copy() + GenerationBatchResult` deque | 基础 generation 分支对齐 |
| request row | NumPy 二维映射，row 0 可分配 | [`ReqToTokenPool`](../../python/sglang/srt/mem_cache/memory_pool.py) | 映射职责对齐；标准版保留 padding row 0，真实请求从 1 开始 |
| KV allocator | slot 0/page 0 padding，page-aware tail reuse | [`allocator/token.py`](../../python/sglang/srt/mem_cache/allocator/token.py)、[`allocator/paged.py`](../../python/sglang/srt/mem_cache/allocator/paged.py) | 核心分配粒度对齐；教学版不保存真实 K/V tensor |
| KV 边界 | `req.kv.kv_allocated_len` / `req.kv_committed_len` | 同名同层级 | 已分配与已提交两条边界对齐 |
| radix cache | page-aligned compressed prefix、lock ref、LRU leaf eviction | [`RadixCache`](../../python/sglang/srt/mem_cache/radix_cache.py) | 基础所有权对齐；无 extra key、host cache、SWA、事件上报 |
| chunked prefill | 唯一 `chunked_req`；当前把非 decode 当作 barrier | 标准 `chunked_req` + `inflight_middle_chunks` | 基础 chunk 生命周期对齐；连续 prefill overlap 是[独立进阶主题](prefill-overlap.md) |
| retract | 释放 row/KV，保留 output，回 waiting 重建 | [`retract_decode()` / `reset_for_retract()`](../../python/sglang/srt/managers/schedule_batch.py) | 基础生命周期对齐；标准版还支持 offload、priority 等 |
| prefill admission | 页对齐成本、decode reserve、chunk/FCFS | [`PrefillAdder`](../../python/sglang/srt/managers/schedule_policy.py) | 职责对齐；教学 `ADMIT/CHUNK/DEFER/ABORT` 与标准 `CONTINUE/NO_TOKEN/OTHER` 不是枚举映射 |
| 请求状态 | `WAITING/PREFILLING/RUNNING/FINISHED` | queues + `finished_reason` + `is_retracted` | 教学状态机，标准 SRT 无同名统一枚举 |
| 延迟释放 | 显式 in-flight owner 计数 | result queue、batch/request 状态与 `release_kv_cache()` | 安全目的对齐，机制不是一一对应 |

## 核心结论

```text
调度：schedule current -> launch current -> FIFO process previous
数据：previous sample -> FutureMap.stash -> successor gather
完成：copy_done -> CPU output_ids / finish
内存：request row -> KV slots/pages -> radix/retract/release
```

这四条核心链路与标准 SRT 对齐。真实 CUDA 执行、复杂 cache、分布式和完整
serving 不属于本教学实现的核心校对范围；连续 prefill overlap 单独讨论。
