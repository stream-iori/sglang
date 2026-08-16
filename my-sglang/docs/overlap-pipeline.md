# Fake CUDA overlap pipeline

本文描述 `my-sglang` 当前实现：CPU 不运行 CUDA，但用确定性的 forward/copy
双逻辑 stream 和 event，对齐 SGLang 基础 generation overlap 的核心因果。

## 1. 先看结论

overlap 不是让同一请求的 B1、B2 forward 并行执行，而是把 CPU 调度/结果处理与
GPU FIFO 执行错开一拍：

```mermaid
flowchart LR
    S1[CPU 调度 B2] --> L2[enqueue B2]
    L2 --> W1[CPU 等待 B1.copy_done]
    W1 --> G1[GPU 执行 B1]
    G1 --> C1[D2H B1 token]
    C1 --> P1[CPU process B1]

    G1 -->|stash token 11| FM[FutureMap row A]
    FM -->|下一轮 gather 11| G2[GPU 执行 B2]

    style S1 fill:#dbeafe,stroke:#2563eb
    style W1 fill:#dbeafe,stroke:#2563eb
    style P1 fill:#dbeafe,stroke:#2563eb
    style G1 fill:#dcfce7,stroke:#16a34a
    style G2 fill:#dcfce7,stroke:#16a34a
    style C1 fill:#fef3c7,stroke:#d97706
    style FM fill:#f3e8ff,stroke:#9333ea
```

关键顺序只有两条：

```text
CPU：launch current B2 -> resolve/process previous B1
GPU：B1 gather/sample/stash -> B2 gather/sample/stash
```

首个 prefill 不能直接进入这条稳定 relay。CPU 必须先 process prefill 结果，让请求
进入 `RUNNING`，之后才能创建第一个 decode batch。

![同步先结算与 overlap 先提交的时间关系](assets/sync-vs-overlap-timeline.png)

图片中的 B2 是“已提交、等待同一条 forward FIFO 的工作”，不是与 B1 同时执行的
第二次 forward。这个区别是理解 overlap 的第一道关：它重叠的是 CPU 调度/提交与 GPU
已入队工作的进度，而不是打破单 stream 的 token 因果。

## 2. 核心模型图

```mermaid
classDiagram
    class MiniOverlapScheduler {
        +result_queue
        +future_map
        -_inflight_refs
        -_deferred_finished
        +pipeline_step()
        +drain()
    }

    class PipelineJob {
        +job_id
        +batch
        +forward
        +result
    }

    class MiniScheduleBatch {
        +reqs
        +forward_mode
        +input_ids
        +req_pool_indices
        +out_cache_loc
        +prepare_for_decode()
        +commit_allocated()
    }

    class ForwardBatch {
        +forward_mode
        +input_ids
        +req_pool_indices
        +out_cache_loc
        +seq_lens
    }

    class FakeGenerationBatchResult {
        +next_token_ids
        +copy_done
        +resolve_cpu_tokens()
    }

    class FakeCudaRunner {
        +forward_stream
        +copy_stream
        +run_batch_async()
        +resolve()
    }

    class FutureMap {
        +output_tokens_buf
        +valid
        +stash()
        +gather()
        +clear()
    }

    class Req {
        +status
        +output_ids
        +req_pool_idx
        +kv.kv_allocated_len
        +kv_committed_len
        +finished_reason
    }

    class ReqToTokenPool {
        +req_to_token
        +free_slots
    }

    MiniOverlapScheduler "1" o-- "0..2" PipelineJob : result_queue
    MiniOverlapScheduler *-- FutureMap
    MiniOverlapScheduler --> FakeCudaRunner
    PipelineJob *-- MiniScheduleBatch
    PipelineJob *-- ForwardBatch
    PipelineJob *-- FakeGenerationBatchResult
    MiniScheduleBatch --> Req
    MiniScheduleBatch --> ReqToTokenPool
    ForwardBatch --> Req
    FakeCudaRunner --> FutureMap : gather / stash
    FutureMap --> ReqToTokenPool : 共用 req_pool_idx 坐标
```

`FutureMap[row]` 和 `ReqToTokenPool[row]` 只是共用稳定的 `req_pool_idx` 坐标，内容
完全不同：

| 对象 | `row` 中保存什么 | 消费者 |
|---|---|---|
| `FutureMap.output_tokens_buf[row]` | 上一个 batch 采样出的 token | 后继 decode 的 forward stream |
| `ReqToTokenPool.req_to_token[row, pos]` | 逻辑 token 位置对应的物理 KV slot | attention / KV 管理 |
| `Req.output_ids` | CPU 已按 FIFO 确认的用户输出 | finish 判断与最终返回 |

## 3. 一次 `pipeline_step()` 的详细流程

```mermaid
flowchart TD
    Start([pipeline_step]) --> Empty{result_queue 为空?}

    Empty -->|是| Fresh[调度 fresh EXTEND 或 DECODE]
    Fresh --> HasFresh{得到 batch?}
    HasFresh -->|是| LaunchFresh[prepare / enqueue / commit KV]
    LaunchFresh --> PushFresh[result_queue.append]
    PushFresh --> ReturnFresh([返回；本轮不 resolve])
    HasFresh -->|否| Admin[处理 retract / abort 等管理结果]
    Admin --> ReturnFresh

    Empty -->|否| Head[读取队首 previous job]
    Head --> Relay{队首请求仍 RUNNING\n且可 relay decode?}
    Relay -->|是| PrepareNext[为 successor 分配 decode KV]
    PrepareNext --> Memory{KV 足够?}
    Memory -->|是| LaunchNext[enqueue successor forward 与 D2H]
    LaunchNext --> PushNext[result_queue.append successor]
    Memory -->|否| Barrier[记录 decode_memory barrier]
    Relay -->|否| Barrier2[记录 prefill / finished barrier]

    PushNext --> ResolveOld[resolve 队首 previous]
    Barrier --> ResolveOld
    Barrier2 --> ResolveOld
    ResolveOld --> Sync[同步 previous.copy_done]
    Sync --> Process[写 output_ids / finish 判断]
    Process --> Pop[result_queue.popleft previous]
    Pop --> Owner[_inflight_refs 减一]
    Owner --> FinishedInflight{successor 已在途\n但请求刚结束?}
    FinishedInflight -->|是| DrainExtra[立即 resolve 多算 successor\n丢弃 token 后安全释放]
    FinishedInflight -->|否| QueueEmpty{result_queue 为空?}
    DrainExtra --> QueueEmpty
    QueueEmpty -->|是| ScheduleAgain[settle 上一批并调度 fresh work]
    QueueEmpty -->|否| Done([一致性检查并返回])
    ScheduleAgain --> Done

    style LaunchFresh fill:#dbeafe,stroke:#2563eb
    style LaunchNext fill:#dbeafe,stroke:#2563eb
    style Sync fill:#fef3c7,stroke:#d97706
    style Process fill:#dbeafe,stroke:#2563eb
    style DrainExtra fill:#fee2e2,stroke:#dc2626
```

`launch` 成功后就提交 KV 水位，因为 forward 已经进入 pipeline；延迟到队首
`copy_done` 的是 CPU token、finish 判断与资源释放。

## 4. B0 到 B3 的完整时序

设请求 A 的 prompt 是 `[1,2]`，`max_new_tokens=3`，模型依次采样
`10,11,12,13`。B0 是 prefill，B1/B2 是有效 decode，B3 是 overlap 提前提交的
多算 decode。

```mermaid
sequenceDiagram
    autonumber
    participant CPU as CPU Scheduler
    participant RQ as result_queue
    participant REQ as Req / KV
    participant FWD as forward_stream
    participant FM as FutureMap[A.row]
    participant CPY as copy_stream

    Note over CPU,REQ: 启动阶段：请求仍是 WAITING
    CPU->>FWD: enqueue B0 prefill + sample
    CPU->>CPY: enqueue B0 D2H + copy_done
    CPU->>RQ: append B0
    CPU->>CPY: synchronize B0.copy_done
    CPY->>FWD: wait B0.forward_done
    FWD->>FM: B0 sample 10；stash 10
    FWD-->>CPY: B0.forward_done
    CPY-->>CPU: D2H token 10；B0.copy_done
    CPU->>REQ: output_ids=[10]；status=RUNNING
    CPU->>RQ: pop B0

    Note over CPU,FM: 第一个 decode 只能在 B0 被 CPU process 后创建
    CPU->>FWD: enqueue B1 decode
    CPU->>CPY: enqueue B1 D2H
    CPU->>RQ: append B1

    Note over CPU,RQ: 稳定 overlap：先提交 B2，再处理 B1
    CPU->>FWD: enqueue B2 decode（排在 B1 后）
    CPU->>CPY: enqueue B2 D2H
    CPU->>RQ: append B2；queue=[B1,B2]
    CPU->>CPY: synchronize B1.copy_done
    CPY->>FWD: wait B1.forward_done
    FWD->>FM: B1 gather 10（consume）
    FWD->>FM: B1 sample 11；stash 11
    FWD-->>CPY: B1.forward_done
    CPY-->>CPU: D2H token 11；B1.copy_done
    CPU->>REQ: output_ids=[10,11]
    CPU->>RQ: pop B1；queue=[B2]

    Note over CPU,RQ: 下一轮仍先提交后继，因此 B3 会提前在途
    CPU->>FWD: enqueue B3 decode（排在 B2 后）
    CPU->>CPY: enqueue B3 D2H
    CPU->>RQ: append B3；queue=[B2,B3]
    CPU->>CPY: synchronize B2.copy_done
    CPY->>FWD: wait B2.forward_done
    FWD->>FM: B2 gather 11（consume）
    FWD->>FM: B2 sample 12；stash 12
    FWD-->>CPY: B2.forward_done
    CPY-->>CPU: D2H token 12；B2.copy_done
    CPU->>REQ: output_ids=[10,11,12]；FINISHED(length)
    CPU->>RQ: pop B2；B3 仍持有请求资源

    Note over CPU,FM: 清理多算 B3，不能提前释放 row/KV
    CPU->>CPY: synchronize B3.copy_done
    CPY->>FWD: wait B3.forward_done
    FWD->>FM: B3 gather 12（consume）
    FWD->>FM: B3 sample 13；stash 13
    FWD-->>CPY: B3.forward_done
    CPY-->>CPU: D2H token 13；B3.copy_done
    CPU->>RQ: pop B3；丢弃 token 13
    CPU->>FM: clear A.row
    CPU->>REQ: owner=0，释放 row / 私有 KV / runner 状态
```

这里能看出三件事：

1. `10` 既进入 CPU 的 `output_ids`，也通过 `FutureMap` 成为 B1 的设备侧输入。
2. `B1 stash(11)` 一定排在 `B2 gather(11)` 前，因为它们位于同一个 forward stream。
3. B3 虽然是多算，也必须执行/退出 result queue 后才能释放请求资源。

## 5. event 如何只推进必要前缀

Fake CUDA 不模拟真实耗时，只模拟 FIFO 和 event 依赖。等待 `B1.copy_done` 时的依赖链：

```mermaid
flowchart LR
    Sync[CPU synchronize\nB1.copy_done] --> CopyRun[copy stream 执行\nB1 D2H]
    CopyRun --> WaitFwd[B1 D2H 等待\nB1.forward_done]
    WaitFwd --> FwdRun[forward stream 执行\nB1 gather/sample/stash]
    FwdRun --> FwdEvent[record\nB1.forward_done]
    FwdEvent --> D2H[复制 B1 token 到 host]
    D2H --> CopyEvent[record\nB1.copy_done]
    CopyEvent --> Resume[CPU 恢复]

    B2[B2 forward 已排队] -. 不属于 B1 event 前缀 .-> Later[后续等待 B2 时执行]

    style Sync fill:#dbeafe,stroke:#2563eb
    style FwdRun fill:#dcfce7,stroke:#16a34a
    style D2H fill:#fef3c7,stroke:#d97706
    style B2 fill:#f3f4f6,stroke:#6b7280
```

因此 `synchronize(B1.copy_done)` 不会顺带执行 B2。真实 CUDA 会自主异步推进，
不需要等 CPU 调用 `synchronize()` 才工作；但相同 stream 的 FIFO 和 event 依赖不变。

把它当作“只补齐到收据所需的账目”：`B1.copy_done` 的依赖只包含 B1 的 forward、
forward event、B1 的 D2H 和 copy event；虽然 B2 已排在后面，它不在这张收据的前缀中。
`test_fake_cuda_event_only_advances_required_forward_prefix` 就断言了这一点。

## 6. 四本账与不变量

| 账本 | 写入时机 | 读取时机 | 核心不变量 |
|---|---|---|---|
| `FutureMap` | forward sample 后 `stash` | 后继 decode `gather` | 每次 gather 必须对应一个更新后的 producer token |
| `result_queue` | batch enqueue 后 | CPU 严格 FIFO resolve/process/pop | 深度最多 2；不能越过队首提交结果 |
| `Req` | CPU process result | 下一轮调度、finish 判断 | `output_ids` 只包含 CPU 已确认 token |
| `_inflight_refs` | job 入队加一，出队减一 | 请求完成或失败清理 | owner 不为 0 时不能释放 row/KV |

```mermaid
stateDiagram-v2
    [*] --> Invalid: row 创建或 gather 后
    Invalid --> Valid: producer stash(token N)
    Valid --> Invalid: successor gather(token N)
    Valid --> Invalid: 请求释放或失败恢复 clear
```

教学版 `FutureMap.valid` 显式实现 consume-once。标准 SRT 生产路径不依赖这个 bool，
但 CI debug 同样会 poison/invalidate 已消费位置，用来发现“没有新 producer 就再次
gather”的错误。

## 7. finish、资源延迟释放与多算

```mermaid
flowchart TD
    Token[CPU process token] --> Finish{EOS / stop /\nmax_new_tokens?}
    Finish -->|否| Running[保持 RUNNING]
    Running --> Next[允许创建后继 decode]
    Finish -->|是| Mark[标记 FINISHED]
    Mark --> Owner{_inflight_refs == 0?}
    Owner -->|是| Release[clear FutureMap\n释放 row / 私有 KV]
    Owner -->|否| Deferred[加入 _deferred_finished]
    Deferred --> Drain[resolve 并移除在途多算 job]
    Drain --> Owner
```

CPU 在 launch B3 时还没看到 B2 的 token 12，所以无法提前知道请求会达到
`max_new_tokens=3`。B2 process 后请求才变成 `FINISHED`；此时 B3 已持有快照、row
和 KV，必须先安全 drain。B3 的 token 13 不会进入 `Req.output_ids`。

## 8. 失败恢复

resolve 或 Fake stream 执行失败时，不能只丢一个队首，否则 FutureMap、KV 水位和
在途 owner 会失去一致性。恢复按整条 pipeline 处理：

```mermaid
flowchart LR
    Fail[resolve / stream 失败] --> Discard[discard 全部在途 result]
    Discard --> ReleasePhysical[释放在途请求的物理 KV 与 row 映射]
    ReleasePhysical --> Clear[clear FutureMap rows]
    Clear --> Keep[保留已由 CPU FIFO 提交的 output_ids]
    Keep --> Waiting[未完成请求 reset 后回 waiting]
    Waiting --> Retry[后续重新 EXTEND 重建上下文]
```

## 9. 与标准 SGLang SRT 的对应

| my-sglang | 标准 SRT | 对齐边界 |
|---|---|---|
| `FutureMap.output_tokens_buf` | 同名字段 | 基础 token relay 对齐 |
| `FutureMap.stash/gather` | `stash` / `resolve_forward_inputs` | 职责对齐；标准版还 relay seq lens 和高级 payload |
| `FakeCudaRunner.forward_stream` | scheduler/model forward stream | FIFO 因果对齐，不模拟真实性能 |
| `FakeCudaRunner.copy_stream` |异步 D2H copy stream | event 依赖对齐 |
| `FakeGenerationBatchResult.copy_done` | `GenerationBatchResult.copy_done` | CPU 可读边界对齐 |
| `result_queue` | `Scheduler.result_queue` | launch current、process previous 的 FIFO 对齐 |
| `pipeline_step()` | `Scheduler.event_loop_overlap()` 主循环 | 基础 generation 分支对齐 |

连续 prefill/chunk overlap 不塞进本章的基础 decode relay，单独见
[prefill overlap](prefill-overlap.md)。speculative decoding 与 grammar 不在本文范围。

## 10. 代码和测试入口

| 阅读顺序 | 文件 | 观察点 |
|---:|---|---|
| 1 | [`runner.py`](../src/my_sglang/runner.py) | stream FIFO、event、gather/stash、D2H |
| 2 | [`overlap_scheduler.py`](../src/my_sglang/overlap_scheduler.py) | enqueue current、finalize previous、owner 计数 |
| 3 | [`test_overlap_scheduler.py`](../tests/test_overlap_scheduler.py) | 顺序、队列深度、finish 多算与失败恢复断言 |
| 4 | [`newcomer-guide.md`](newcomer-guide.md) | 单请求 trace 的逐步解释 |
