# my-sglang 源码导读

这份导读只回答两个问题：先读什么，以及一条请求实际经过哪些方法。
机制细节继续看 [Scheduler / KVCache 核心结构](scheduler-kv-overview.md)。

## 1. 先记住四层

```text
请求状态层      models.py
    Req / BatchForward
         │
         ▼
索引与内存层    pools.py + radix_cache.py
    request row → seq_pos → KV slot → page
         │
         ▼
单批事务层      schedule_batch.py + schedule_policy.py
    admission → prepare → commit / rollback
         │
         ▼
调度执行层      scheduler.py + overlap_scheduler.py + runner.py
    选批 → forward → 处理 token → 进入下一轮
```

| 层 | 先回答的问题 | 推荐入口 |
|---|---|---|
| 请求状态 | 一个请求跨轮保存什么？ | `models.Req` |
| 内存索引 | token 如何找到 KV？ | `ReqToTokenPool.write()` |
| Batch | 本轮申请了什么？ | `MiniScheduleBatch.prepare_for_extend()` |
| Admission | 请求为什么能进/不能进？ | `PrefillAdder._decide()` |
| Scheduler | 一轮按什么顺序执行？ | `MiniScheduler.step()` |
| Overlap | launch 与 finalize 为什么拆开？ | `MiniOverlapScheduler.launch_step()` |

## 2. 普通请求主线

```text
add_request
    │
    ▼
waiting_queue
    │
    ▼
_get_new_prefill_batch
    ├─ PrefillAdder.add_requests       只做预算决策
    ├─ _attach_new_request             绑定 request row / cache prefix
    └─ prepare_for_extend              分配 suffix KV slot
    │
    ▼
runner.prefill / runner.extend
    │
    ├─ prompt 未完成 → chunked_req
    └─ prompt 已完成 → running_batch
                          │
                          ▼
                    _get_decode_batch
                          │
                          ▼
                    runner.decode_batch
                          │
                  EOS / length / 下一轮
```

建议在以下位置打断点：

1. `MiniScheduler.step()`：观察本轮选择 EXTEND 还是 DECODE。
2. `MiniScheduleBatch.prepare_for_extend()`：观察 allocated 前进。
3. `MiniScheduleBatch.commit_allocated()`：观察 committed 追平。
4. `MiniScheduler._process_batch_result()`：观察请求状态变化。

## 3. 三种执行入口

| 入口 | 时间线 | 用途 |
|---|---|---|
| `MiniScheduler.step()` | prepare → 同步 forward → commit | 先学基础生命周期 |
| `launch_step()` + `finalize_pending()` | prepare/start/kick → finalize/commit | 观察事务窗口 |
| `pipeline_step()` | launch B1 → finalize B0 | 学设备侧 future 和 FIFO 提交 |

### Manual overlap

```text
launch_step
    allocate KV → start/kick → _pending
                                  │
                     allocated > committed
                                  │
finalize_pending                  ▼
    materialize token → commit → process Req → clear _pending
```

### Production-shaped pipeline

```text
turn N       queue = [B0]
                 │
turn N+1     launch chained B1
                 │
             queue = [B0, B1]
                 │
             finalize/process B0
                 │
             queue = [B1]
```

设备计算可以提前，但 `Req.output_ids`、finish 和资源释放必须按 FIFO 提交。

## 4. 最容易混淆的名字

| 名字 | 它是什么 | 它不是什么 |
|---|---|---|
| `Req` | 跨多轮存在的请求状态 | 单次 forward 参数 |
| `MiniScheduleBatch` | 本轮可变工作单 | 长期 KV 容器 |
| `BatchForward` | runner 的不可变参数快照 | 可推进的状态机 |
| `running_batch` | 下一轮 decode 候选集合 | 当前一定正在执行的 batch |
| `last_batch` | 上一轮到下一轮的交接站 | 历史记录 |
| `prefix_len` | 已有可用 KV 的 token 边界 | KV slot id |
| `kv_allocated_len` | 已预留物理 slot 的边界 | 已确认模型成功的边界 |
| `kv_committed_len` | 模型已成功计算的 KV 边界 | allocator 总容量 |

## 5. 推荐测试顺序

```bash
cd my-sglang

# 1. 行映射与 page
../python/.venv/bin/python -m pytest tests/test_pools.py -q

# 2. Admission
../python/.venv/bin/python -m pytest tests/test_schedule_policy.py -q

# 3. 普通生命周期
../python/.venv/bin/python -m pytest tests/test_scheduler.py -q

# 4. Radix cache
../python/.venv/bin/python -m pytest tests/test_radix_cache.py -q

# 5. Overlap
../python/.venv/bin/python -m pytest tests/test_overlap_scheduler.py -q
```
