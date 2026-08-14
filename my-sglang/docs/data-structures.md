# 数据结构、所有权与不变量

先记住两条链：请求状态从 `Req` 进入可变 `MiniScheduleBatch`，最后变成给 runner 的不可变 `ForwardBatch`；KV 地址从 request row 和逻辑位置映射到 slot，再由 allocator 解释 slot 属于哪个 page。

```mermaid
flowchart LR
    Req[Req] --> MB[MiniScheduleBatch<br/>mutable scheduling state]
    MB --> BF[ForwardBatch<br/>immutable runner view]
    Req -->|req_pool_idx| Row[ReqToTokenPool row]
    Row -->|seq_pos| Slot[KV slot]
    Slot --> Page[allocator page]
    Cache[MiniRadixCache] -->|owns / lends| Slot
    Req -. last_node lock .-> Cache
```

<a id="req-state"></a>
## 1. `Req`：逻辑 token 与物理 KV 边界

[`Req`](../src/my_sglang/models.py) 跨多个 step 存活。关键字段不是 slot 列表，而是长度边界和 `ReqToTokenPool` 行。

| 字段                                | 含义                                                                 | 主要修改方法                                                       |
| --------------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------ |
| `origin_input_ids` / `output_ids` | 固定 prompt / 已向用户确认的生成 token                                        | [`append_output()`](../src/my_sglang/models.py)              |
| `get_fill_ids()`                  | prompt；retract 后为 prompt + 已生成 token，用来重建上下文                       | [`reset_for_retract()`](../src/my_sglang/models.py)          |
| `req_pool_idx`                    | 二维映射矩阵中的活跃行                                                        | [`_attach_new_request()`](../src/my_sglang/scheduler.py)     |
| `extend_range`                    | 本轮 EXTEND 的请求内 `[start,end)` 区间                                       | [`_get_new_prefill_batch()`](../src/my_sglang/scheduler.py)  |
| `kv.kv_allocated_len`             | 已有物理 slot；forward 可能尚未成功                                           | [`prepare_for_extend()`](../src/my_sglang/schedule_batch.py) |
| `kv_committed_len`                | scheduler 已接受的 KV 边界；同步路径在 runner 返回后提交，overlap 路径在异步 launch 成功后提交 | [`commit_allocated()`](../src/my_sglang/schedule_batch.py)   |
| `prefix_indices` / `last_node`    | 从 radix cache 借用的 slot / 被锁住的终点节点                                  | [`_attach_new_request()`](../src/my_sglang/scheduler.py)     |
| `cache_protected_len`             | 当前请求锁住的 cache prefix 长度                                            | [`_cache_unfinished_req()`](../src/my_sglang/scheduler.py)   |
| `retracted_stain`                 | 曾被 retract；再次 admission 时预留全部剩余输出                                  | [`_retract_req()`](../src/my_sglang/scheduler.py)            |

### 与标准 SGLang 的 `Req` 全字段对照

标准 SGLang 的 [`Req`](../../python/sglang/srt/managers/schedule_batch.py) 包含同一条
请求/KV/prefix 主链，但运行在真实设备 KV 和更完整的 serving 场景中。下表的“对应”
表示职责可对照，不表示两个对象可替换，也不表示生命周期完全一样。

下面覆盖 [`my_sglang.models.Req`](../src/my_sglang/models.py) 的全部 dataclass
字段。标记“同名”时，可以直接拿字段名去标准 SRT 搜索；标记“职责对应”时，必须
注意对象层级或表示方式已经变化。

| my-sglang `Req` 字段 | 标准 SRT 当前字段/接口 | 关系 | 阅读标准代码时怎么理解 |
|---|---|---|---|
| `rid` | `Req.rid` | 同名 | 请求唯一标识 |
| `origin_input_ids` | `Req.origin_input_ids` | 同名 | 原始 prompt；标准版另有 `origin_input_ids_unpadded` |
| `sampling_params` | `Req.sampling_params` | 同名 | 标准类型位于 `srt/sampling/sampling_params.py`，配置更完整 |
| `eos_token_ids` | `Req.eos_token_ids` | 同名 | 模型 EOS 集合；不同于用户配置的 `sampling_params.stop_token_ids` |
| `output_ids` | `Req.output_ids` | 同名 | 都是 append-only 生成结果；标准版使用 `array("q")` |
| `status` | 无统一字段 | 教学专用 | 标准版由 waiting/running/chunked 容器、`finished_reason`、`is_retracted` 共同表达状态 |
| `req_pool_idx` | `Req.req_pool_idx` | 同名 | `(request row, sequence position) -> KV slot` 的稳定 row；标准 row 0 保留作 padding |
| `prefix_indices` | `Req.prefix_indices` | 同名 | cache 命中的设备 KV slot 序列；教学版为 NumPy |
| `last_node` | `Req.last_node` | 同名 | radix cache 当前锁住路径的终点节点 |
| `cache_protected_len` | `Req.cache_protected_len` | 同名 | 已交给 cache 且仍受请求保护的 prefix 长度 |
| `extend_range` | `Req.extend_range` | 同名同类型 | 都使用带 `start/end/length` 的 `Range` 表示本轮 EXTEND 区间 |
| `kv` | `Req.kv` | 同名同层级 | 都是 `ReqKvInfo`；教学版不实现 SWA，因而省略 `swa_evicted_seqlen` |
| `kv.kv_allocated_len` | `Req.kv.kv_allocated_len` | 同名同层级 | 已经分配物理 KV 的边界；教学 `ReqKvInfo` 只保留这一核心字段 |
| `kv_committed_len` | `Req.kv_committed_len` | 同名 | scheduler 已提交、可作为稳定历史使用的 KV 边界 |
| `retracted_stain` | `Req.retracted_stain` | 同名 | 是否曾 retract，用于更保守的再次 admission |
| `finished_reason` | `Req.finished_reason` | 同名同结构 | 都使用 `FINISH_MATCHED_TOKEN` / `FINISH_LENGTH` / `FINISH_ABORT` 对象 |

`SamplingParams` 的两个教学字段也要分开看：

| my-sglang | 标准 SRT | 说明 |
|---|---|---|
| `max_new_tokens` | `SamplingParams.max_new_tokens` | 同名 |
| `Req.eos_token_ids` | `Req.eos_token_ids` | 同名；模型 EOS 集合 |
| `SamplingParams.stop_token_ids` | `SamplingParams.stop_token_ids` | 同名；用户指定的 stop-token 集合 |

`Req` 的计算属性/方法不是额外状态，对照如下：

| my-sglang 计算属性 | 标准 SRT 对应写法 |
|---|---|
| `generated_count` | `len(req.output_ids)` |
| `full_untruncated_fill_ids` | `req.full_untruncated_fill_ids`；同名基础 generation 视图 |
| `get_fill_ids()` | `req.get_fill_ids()`；同名 |
| `last_token_id` | `req.get_fill_ids()[-1]`；overlap decode 通常直接来自 FutureMap |
| `remaining_new_tokens` | `req.sampling_params.max_new_tokens - len(req.output_ids)` |

```text
my-sglang:   保留 scheduler、slot、page、prefix 所有权这一条主干
标准 SGLang: 主干 + 真实 GPU KV + 多模态 + session + 分层 cache + 分布式/复杂调度
```

阅读标准实现时，可从 `Req` 的输入/KV 字段、prefix 字段和
[`get_fill_ids()`](../../python/sglang/srt/managers/schedule_batch.py) 三处回照本节；
不要把标准版新增字段倒灌进教学模型，除非要专门教学它解决的那个问题。

教学版的 `RequestStatus` 是显式状态机；标准 SRT 主要通过 waiting/running/
chunked 容器、`finished_reason`、`is_retracted` 等共同表示生命周期，没有一个
与此枚举一一对应的字段。

核心边界始终满足：

```text
0 <= cache_protected_len <= kv_committed_len <= kv.kv_allocated_len
req_to_token[row, :kv.kv_allocated_len] 都是 allocator 当前拥有的 slot
同步路径中，刚生成的最后一个 output token 尚未进入 KV
overlap 若已提交后继 batch，KV 边界可领先 CPU 已提交的 output_ids 一轮
```

### 用一个请求把这些长度分开

假设 `prompt=[7,8]`，已经向调用方返回 `10`，并且下一轮 decode 已经成功
enqueue 到 Fake CUDA。此时不要把三个“长度”当成同一个概念：

| 观察项 | 值 | 为什么 |
|---|---:|---|
| `full_untruncated_fill_ids` | `[7,8,10]` | `10` 已经是确认输出 |
| `kv_committed_len` | `3` | prompt `[7,8]` 和 decode 输入 `10` 已成功 enqueue |
| `kv.kv_allocated_len` | `3` | 本轮为输入 token `10` 预留的 slot 已提交 |
| `req_to_token[row, :3]` | `[2,3,4]` | 三个 slot 都已提交；只有完整 page 才能作为可复用 cache 前缀 |

```text
逻辑 token:        [7, 8, 10]
KV 已提交:          [7, 8, 10]
                     ^ committed=allocated=3
```

`commit_allocated()` 在 launch 成功后把 committed 推到 3。若 launch 失败，
`rollback_uncommitted()` 清掉新分配的位置 2 的映射，并把 `allocated` 拉回 2。Fake CUDA
pipeline 的关键是：KV 已提交不代表 `11` 已写入 `output_ids`；后者仍要等待
result queue 队首的 `copy_done`。

状态机：

```mermaid
stateDiagram-v2
    [*] --> WAITING
    WAITING --> PREFILLING: 非最后 chunk
    PREFILLING --> PREFILLING: 后续非最后 chunk
    WAITING --> RUNNING: 完整 EXTEND
    PREFILLING --> RUNNING: 最后 chunk
    RUNNING --> WAITING: decode retract
    WAITING --> FINISHED: admission abort / EXTEND stop
    RUNNING --> FINISHED: decode stop / OOM abort
```

### 状态机怎么读：状态描述请求，`EXTEND` / `DECODE` 描述这一步 batch

最容易混淆的是把 `PREFILLING` 当作“只要在做 prefill 就会处于这个状态”。
这里不是这样：`RequestStatus` 是 `Req` 跨多个 `step()` 的状态；
`ForwardMode` 是**本次** runner 调用的模式。一个没有分块的普通 prompt 会经历一次
`EXTEND`，但结果立刻进入 `RUNNING`，根本不会停在 `PREFILLING`。

| 请求状态 | 人话含义 | 这一步可被调度成什么 batch | 进入条件 | 离开条件 / 持有资源 |
|---|---|---|---|---|
| `WAITING` | 排队等 admission，或被撤回后等待重建 KV | `EXTEND` | 新建 `Req` 的初始状态；或者 decode 内存紧张时 `retract` | admission 成功：完整 prompt 直接到 `RUNNING`，非最后 chunk 到 `PREFILLING`；admission 判定物理上不可能时到 `FINISHED`。新请求通常还没有 row/KV；retract 后逻辑 token 保留，但 row、非缓存 KV 与 runner 状态已释放。 |
| `PREFILLING` | 长 prompt 正在分块填充，尚不能生成/返回第一个 token | 下一次仍是 `EXTEND`，且该请求是唯一 `chunked_req` | 一个 `EXTEND` chunk 成功，但它不是 `get_fill_ids()` 的最后一段 | 后续 chunk 仍不完整则留在本状态；最后 chunk 成功后，拿到第一个 output，转 `RUNNING` 或（EOS/长度到达）`FINISHED`。已完成的 KV 会保留；只有完整 page 可提前放入并锁住 radix cache。 |
| `RUNNING` | prompt 已处理完，已拿到第一个 output；之后逐 token decode | `DECODE`（每个请求把“上一个 output token”写入 KV） | 完整 `EXTEND` 成功且尚未停止；或最后一个 chunk 成功 | decode 生成 EOS / 达到 `max_new_tokens`：`FINISHED`；decode 内存不足且被选为 victim：回到 `WAITING`，以后以 `prompt + 已生成 output` 重新 `EXTEND`。持有 active row、已确认 KV，以及可能锁住的 cache prefix。 |
| `FINISHED` | 终态，不再参与调度 | 不会进入 batch | EOS、长度上限、admission abort，或最后一个无法回收 KV 的请求 OOM abort | 无后继状态。正常完成时可缓存完整 page，但 active row、请求私有 KV、runner 状态都会释放。 |

可以把状态转换压缩成下面这四句话：

```text
短 prompt：       WAITING --一次完整 EXTEND--> RUNNING --DECODE...--> FINISHED
长 prompt：       WAITING --非最后 EXTEND--> PREFILLING --最后 EXTEND--> RUNNING
decode 内存紧张： RUNNING --retract（释放物理状态）--> WAITING --重新 EXTEND--> RUNNING
不能执行：        WAITING / RUNNING -------------------------------> FINISHED
```

`PREFILLING` 的关键不是“正在算 prompt”，而是“**prompt 还没有全部写进 KV，因而还不能产生第一个可返回 token**”。
`RUNNING` 的关键不是“GPU 此刻正在运行”，而是“请求已经具备逐 token decode 的资格”。

### 例子 A：短请求为什么跳过 `PREFILLING`

以下表描述同步 scheduler。设 `prompt=[7,8]`，`max_new_tokens=2`，runner 在 prefill 后返回 `10`，下一轮 decode 返回
`11`。未启用 chunked prefill，所以 prompt 一次就能处理完。

| `step()` | 调度的 `ForwardBatch.forward_mode` 与输入 | runner 新返回的 token | 请求状态（本 step 结束时） | `output_ids` | KV 边界（本 step 成功后） |
|---:|---|---:|---|---|---|
| 0（刚入队） | 无 | 无 | `WAITING` | `[]` | `allocated=committed=0` |
| 1 | `EXTEND`，输入 `[7,8]` | `10` | `RUNNING` | `[10]` | `[7,8]` 已写入 KV，`allocated=committed=2` |
| 2 | `DECODE`，输入 `[10]` | `11` | `FINISHED`（达到 2 个输出） | `[10,11]` | `[10]` 也已写入 KV，随后 active KV/row 被释放 |

注意 decode 的因果关系：第 2 步的输入是已在第 1 步返回的 `10`，模型在它的 KV
上下文上预测出 `11`。因此结束时最后生成的 `11` 已交给用户，却还没有被写入 KV；如果
它没有结束，请求会保持 `RUNNING`，下一次 `DECODE` 才把 `11` 作为输入写入 KV。

### 例子 B：长请求如何停在 `PREFILLING`

设 `prompt=[1,2,3,4,5]`，`chunked_prefill_size=2`，`max_new_tokens=2`。runner 对前两段
只完成 KV；只有最后一段才返回第一个 output。

| `step()` | `EXTEND` 输入 / 是否最后 chunk | `extend_range.end=kv_committed_len` | 状态（本 step 结束时） | `output_ids` | 为什么 |
|---:|---|---:|---|---|---|
| 1 | `[1,2]` / 否 | 2 | `PREFILLING` | `[]` | prompt 还有 `[3,4,5]`，返回的值不能作为用户 output。 |
| 2 | `[3,4]` / 否 | 4 | `PREFILLING` | `[]` | prompt 还剩 `[5]`；若 page size 为 2，此时 `[1,2,3,4]` 可作为完整 cache page。 |
| 3 | `[5]` / 是 | 5 | `RUNNING` | `[10]` | prompt 已完整处理，prefill 的返回值 `10` 才是第一个生成 token。 |
| 4 | `DECODE` 输入 `[10]` | 6 | `FINISHED` | `[10,11]` | decode 返回第二个 token `11`，达到长度上限。 |

如果第 3 步后发生 decode OOM，调度器可能选择这个请求做 `retract`：状态回到
`WAITING`，但 `output_ids=[10]` 不会丢。下次 admission 时，`get_fill_ids()` 已变为
`[1,2,3,4,5,10]`，它会重新走 `EXTEND` 来恢复上下文，而不是重新生成 `10`。

<a id="batch-forward"></a>
## 2. `FutureMap` 与 result queue：overlap 的两本新账

同步 scheduler 只要处理“本轮输入、KV、输出”。overlap 多出两本账：一份让下一轮
forward 继续跑，一份让 CPU 按顺序晚点提交结果。

```text
row 3 的两个独立位置

FutureMap.output_tokens_buf[3]  = 11   # B2 的设备侧输入
ReqToTokenPool.req_to_token[3]  = ...  # A 的 KV slot 映射
```

| 对象 | 保存什么 | 何时写入 | 何时读取/清理 |
|---|---|---|---|
| `FutureMap` | 下一轮 decode 的 token 值与 valid bit | forward stream 的 sampling 后 | 后继 decode gather；无后继 relay 的非 `RUNNING` 结果、失败恢复或请求释放时 clear |
| `result_queue` | `ForwardBatch` 快照和 `FakeGenerationBatchResult` | 当前 batch enqueue 后 | CPU 严格 FIFO resolve/process/pop |
| `copy_done` | host buffer 是否可读的 event | copy stream 的 D2H 后 | 只在处理 queue 队首时 synchronize |

`FutureMap` 的 key 是稳定的 `req_pool_idx`，不是会随 batch 重排的 batch 下标。详情见 [overlap 流水线](overlap-pipeline.md)。
教学版的 valid bit 和 `clear()` 是显式安全账，并在 gather 后立即失效；这对应
标准 SRT CI debug 的 consume-once 检查。标准生产路径不维护这个 bool，也不依赖
请求释放时 clear token buffer。

## 3. `MiniScheduleBatch` 与 `ForwardBatch`

[`MiniScheduleBatch`](../src/my_sglang/schedule_batch.py) 是 scheduler 内部的可变对象，负责分配、映射、提交、回滚和 batch 过滤；[`ForwardBatch`](../src/my_sglang/models.py) 是调用 runner 前生成的只读快照。

| 阶段 | 可变状态 | 方法 |
|---|---|---|
| schedule | `reqs / forward_mode / extend_range` 已确定 | [`_get_new_prefill_batch()`](../src/my_sglang/scheduler.py) |
| allocate | 写 `req_to_token`，推进 `kv.kv_allocated_len` | [`prepare_for_extend()`](../src/my_sglang/schedule_batch.py)、[`prepare_for_decode()`](../src/my_sglang/schedule_batch.py) |
| snapshot | 生成 runner 所需 tuple | [`to_forward_batch()`](../src/my_sglang/schedule_batch.py) |
| success | `committed = allocated` | [`commit_allocated()`](../src/my_sglang/schedule_batch.py) |
| failure | 清掉 committed 后的映射，只释放不共享的 page | [`rollback_uncommitted()`](../src/my_sglang/schedule_batch.py) |
| next step | finished/chunked 过滤，或 merge 到 running | [`_settle_last_batch()`](../src/my_sglang/scheduler.py) |

`ForwardBatch` 的第 0 维总与 `reqs` 对齐：

| 字段 | EXTEND | DECODE |
|---|---|---|
| `input_ids`（展平） | 未缓存 suffix / 当前 chunk | 每个请求最后一个逻辑 token |
| `out_cache_loc`（展平） | 本轮 suffix 的 slot | 每请求一个输入 slot |
| `seq_lens` | 当前 fill 终点 | 本轮输入写入后的长度 |
| `prefix_indices_by_req` | radix 命中 | 通常为空 |
| `extend_seq_lens` | suffix 长度 | 全 1 |

### `MiniScheduleBatch` 全字段对照

标准 SRT 把 scheduler 工作单也叫 `ScheduleBatch`。教学版保留 `Mini` 前缀，避免
误认为它包含完整设备张量。以下覆盖教学 dataclass 的全部字段：

| my-sglang `MiniScheduleBatch` | 标准 SRT 当前字段/接口 | 关系 / 形状差异 |
|---|---|---|
| `reqs` | `ScheduleBatch.reqs` | 同名、同职责 |
| `forward_mode` | `ScheduleBatch.forward_mode` | 同名；都区分 EXTEND/DECODE |
| `req_to_token_pool` | `ScheduleBatch.req_to_token_pool` | 同名；NumPy 对应设备 tensor pool |
| `token_to_kv_pool_allocator` | `ScheduleBatch.token_to_kv_pool_allocator` | 同名、同职责 |
| `tree_cache` | `ScheduleBatch.tree_cache` | 同名；教学版只实现基础 radix cache |
| `chunked_req` | `ScheduleBatch.chunked_req` | 同名；当前 batch 的 chunked 请求引用 |
| `first_extend_by_req` | 无直接字段 | 教学 runner 用来区分首次 `prefill()` 和后续 `extend()`；标准 runner 统一按 EXTEND metadata 执行 |
| `input_ids` | `ScheduleBatch.prefill_input_ids_cpu` / `input_ids` | 主字段展平对齐；教学版 flat tuple，`input_ids_by_req` 只是派生观察属性 |
| `req_pool_indices` | `ScheduleBatch.req_pool_indices` | 同名；标准版另有 CPU mirror `req_pool_indices_cpu` |
| `out_cache_loc` | `ScheduleBatch.out_cache_loc` | 同名同坐标系；教学版 flat NumPy array，`out_cache_loc_by_req` 只是派生观察属性 |
| `seq_lens` | `ScheduleBatch.seq_lens` | 同名；每个请求本轮结束后的逻辑长度 |
| `prefix_lens` | `ScheduleBatch.prefix_lens` | 同名；教学版取已分配 KV 起点，标准版取本轮 cache/prefix 边界；基础路径数值相同 |
| `extend_lens` | `ScheduleBatch.extend_lens` / `Req.extend_range.length` | 同名 batch 字段；同时可由每个请求的 range 得到 |

### `ForwardBatch` 全字段对照

教学 `ForwardBatch` 是冻结快照；标准 SRT 的边界分成 scheduler 的
`ScheduleBatch.copy()` 和 model runner 的 `ForwardBatch`，所以不总是一字段对应一字段。

| my-sglang `ForwardBatch` | 标准 SRT 当前字段/接口 | 关系 / 形状差异 |
|---|---|---|
| `forward_mode` | `ForwardBatch.forward_mode` | 同名、同职责 |
| `reqs` | `ScheduleBatch.reqs` | runner 前的 batch 快照仍持有请求；标准 `ForwardBatch` 不把完整 `Req` 当核心输入 |
| `input_ids` | `ForwardBatch.input_ids` | 同名同坐标系；教学版 flat tuple，标准版 flat GPU tensor；`input_ids_by_req` 只是派生观察属性 |
| `req_pool_indices` | `ForwardBatch.req_pool_indices` | 同名、同职责 |
| `out_cache_loc` | `ForwardBatch.out_cache_loc` | 同名同坐标系；教学版 flat tuple，标准版 flat tensor；`out_cache_loc_by_req` 只是派生观察属性 |
| `seq_lens` | `ForwardBatch.seq_lens` | 同名、同职责 |
| `prefix_indices_by_req` | `Req.prefix_indices` + `ReqToTokenPool.req_to_token` | 主名对齐 `Req`；标准 `ForwardBatch` 不携带同形状字段 |
| `extend_seq_lens` | `ForwardBatch.extend_seq_lens` | 同名；标准这里的 `seq` 指本轮 query/extend 长度 |
| `extend_range_starts` | `Req.extend_range.start` | 名字明确使用请求内坐标；不要误映射到 batch-flat 的 `extend_start_loc` |
| `contains_last_prefill_chunk` | `ScheduleBatch.contains_last_prefill_chunk` | 同名；教学快照为便于观察而保留，标准字段位于 `ScheduleBatch` |
| `seq_lens_sum` | `ForwardBatch.seq_lens_sum` | 同名；教学版由 `seq_lens` property 计算 |

最容易误读的两个字段：

```text
my extend_range_starts       = 每个请求内部的绝对 token 起点
SRT ForwardBatch.extend_start_loc = 各请求 EXTEND 数据在展平 input tensor 中的起点
```

它们都叫“start”，但坐标系不同，不能为了同名而强行合并。

### 其余核心对象字段速查

下面补齐从 scheduler 继续追到 pool、cache 和 overlap 时会遇到的字段。以下划线
开头且只为测试/trace 服务的字段会明确标为教学专用。

| my-sglang 对象与字段 | 标准 SRT 对应 | 关系 |
|---|---|---|
| `ReqToTokenPool.size` | `ReqToTokenPool.size` | 同名；可用真实请求行数，标准另加 padding row 0 |
| `ReqToTokenPool.max_context_len` | 同名 | 每行最大逻辑长度 |
| `ReqToTokenPool.req_to_token` | 同名 | 核心二维 row→KV-slot 映射 |
| `ReqToTokenPool.free_slots` | `ReqToTokenPool.free_slots` | 同名；空闲请求行 |
| `ReqToTokenPool._rid_to_idx` | 无 | 教学版防重复和可观测索引；标准直接由 `Req.req_pool_idx` 持有关系 |
| allocator `size` | allocator `size` | 同名；可管理 token slot 数 |
| allocator `page_size` | 同名 | page 粒度；非分页 allocator 为 1 |
| allocator `num_pages` | 由 `size/page_size` 得到 | 教学版显式缓存 |
| allocator `free_pages` | allocator `free_pages` | 同名；空闲 page/slot 索引 |
| allocator `_allocated_pages` | 无独立同名集合 | 教学断言账；标准主要从 free/allocated tensor 与 cache 所有权推导 |
| `MiniRadixCache.root_node` | `RadixCache.root_node` | 同名；radix 哨兵根节点 |
| `MiniRadixCache.page_size` | `RadixCache.page_size` | 同名 |
| `MiniRadixCache.max_slots` | 无直接字段 | 教学版人为 cache 容量；标准回收预算来自真实 allocator/cache 大小 |
| `TreeNode.key` | `TreeNode.key` (`RadixKey`) | 同名；教学版直接用 token tuple，标准版封装 `RadixKey` |
| `TreeNode.value` | `TreeNode.value` | 同名；教学版 slot tuple，标准版 tensor |
| `TreeNode.parent/children` | `TreeNode.parent/children` | 同名树结构 |
| `TreeNode.last_access_time` | `TreeNode.last_access_time` | 同名 LRU 时间 |
| `TreeNode.lock_ref` | `TreeNode.lock_ref` | 同名；大于 0 时节点不可淘汰 |
| `FutureMap.output_tokens_buf` | `FutureMap.output_tokens_buf` | 同名核心 token relay buffer |
| `FutureMap.valid` | CI 下 buffer 的 `-1` poison/invalidate | 教学版显式 bool；标准生产路径无此字段 |
| `FakeGenerationBatchResult.next_token_ids` | `GenerationBatchResult.next_token_ids` | 同名；D2H 前后都沿用该字段，和标准 `copy_to_cpu()` 一致 |
| `FakeGenerationBatchResult.copy_done` | `GenerationBatchResult.copy_done` | 同名 D2H 完成 event |
| `MiniScheduler.waiting_queue` | `Scheduler.waiting_queue` | 同名 |
| `MiniScheduler.running_batch` | `Scheduler.running_batch` | 同名 |
| `MiniScheduler.chunked_req` | `Scheduler.chunked_req` | 同名 |
| `MiniScheduler.last_batch` | `Scheduler.last_batch` | 同名 |
| `MiniScheduler.req_to_token_pool` | `Scheduler.req_to_token_pool` | 同名 |
| `MiniScheduler.token_to_kv_pool_allocator` | `Scheduler.token_to_kv_pool_allocator` | 同名 |
| `MiniScheduler.tree_cache` | `Scheduler.tree_cache` | 同名 |
| `MiniScheduler.max_prefill_tokens` | `Scheduler.max_prefill_tokens` | 同名预算上限 |
| `MiniScheduler.chunked_prefill_size` | `Scheduler.chunked_prefill_size` | 同名 chunk 上限 |
| `MiniScheduler.new_token_ratio` | `Scheduler.new_token_ratio_tracker.current` / `PrefillAdder.new_token_ratio` | scheduler 当前值在 tracker 内；传给 adder 后同名 |
| `MiniOverlapScheduler.result_queue` | `Scheduler.result_queue` | 同名；同一 FIFO |
| `MiniOverlapScheduler.future_map` | `Scheduler.future_map` | 同名；同一 relay 对象 |
| `_inflight_refs` / `_deferred_finished` | 无一一对应字段 | 教学版显式 owner 账；标准由 result queue、batch/Req 状态和释放条件共同保证 |

准入结果不要按枚举名称硬对齐：

| my-sglang | 标准 SRT | 正确读法 |
|---|---|---|
| `AdmissionDecision` | 无同形状公开对象 | 教学版把一次 `PrefillAdder` 决策冻结下来便于测试 |
| `ADMIT/CHUNK/DEFER/ABORT` | `CONTINUE/NO_TOKEN/OTHER` + adder 状态 | 对齐“继续加入还是停止”的职责，不是枚举值映射 |
| `MemoryBudget.remaining_prefill_tokens` | `PrefillAdder.rem_input_tokens` | 本轮剩余 prefill token 预算 |
| `MemoryBudget.remaining_tokens` | `PrefillAdder.rem_total_tokens` 一类动态预算 | 职责对应；标准还叠加 page、cache 和运行请求 offset |

因此，阅读标准 SRT 时建议按下面的名字跳转：

```text
Req.extend_range
  -> ScheduleBatch.prepare_for_extend()
  -> ForwardBatch.extend_prefix_lens / extend_seq_lens
  -> ReqToTokenPool.req_to_token
  -> token_to_kv_pool_allocator
  -> FutureMap.output_tokens_buf（下一轮 decode）
```

## 4. `ReqToTokenPool`：固定二维 NumPy 映射

[`ReqToTokenPool`](../src/my_sglang/pools.py) 的 `req_to_token` 是形状 `(max_running_reqs, max_context_len)` 的 `np.int64` 数组，`-1` 表示未映射。

以 page size 2、请求行 0、prompt `[7,8,9]` 为例：

```text
page 0: slots [0,1]    reserved padding, never allocated
page 1: slots [2,3]    token 7,8
page 2: slots [4,5]    token 9, free tail

req_to_token[0, :3] = [2,3,4]
```

下一轮 decode 输入可直接复用 page 2 的尾 slot 5，不申请新 page。再下一轮才申请 page 3 的 slot 6。该行为由 [`test_paged_allocator_reuses_tail_before_allocating_next_page`](../tests/test_pools.py) 固定。

allocator 的 `available_size/allocated_size` 按完整 page 计数，因此已分配 page 的空尾 slot 不会出现在 `available_size`，只能通过 `last_loc` 被同一序列继续利用。

例如仍取 `page_size=2`，请求 A 的三个 token 已占用 slot `[2,3,4]`：

| page | slot | 当前归属 | 对 `available_size` 的影响 |
|---|---|---|---|
| page 1 | `[2,3]` | A 已写满 | 整页已分配，不可用 |
| page 2 | `[4,5]` | A 使用 `4`；`5` 是 A 的尾部空位 | page 2 已归 A，故 `5` **不**计入 available |
| page 3 | `[6,7]` | 完整空闲 | 贡献 `2` 个 available slot |

所以此刻 `available_size=2`，对应的是完整的 page 3，而不是“所有没写 token 的位置”。
若 A 下一轮 decode，scheduler 通过 `last_loc=4` 知道 A 的最后一个 slot 在 page 2，便可把
`5` 接着分给 A，且 `allocated_size` 仍不变；若是新请求 B，则必须从完整空闲的 page 3
拿 `[6,7]`，不能借 A 的 `5`。这是因为 allocator 按 page 分配/释放：让 A、B 共用 page 2，
将来释放 A 或 B 时就会错误地释放对方仍在使用的 page。

### 三种“空闲”不要混淆

| 名称 | 例子 | 能否立刻给新请求使用 |
|---|---|---|
| allocator free page | page 3 从未分配或已完整释放 | 能 |
| 已分配 page 的尾 slot | page 2 的 slot 5 | 只能给持有 page 2 的同一请求续写 |
| radix evictable slot | cache 中无 lock 的完整 page | 先 LRU evict，再由 scheduler free，随后才能使用 |

所以“`available_size=0`”不必然表示完全无法执行：同一请求可能还能复用尾页；
反过来，“cache 有 slot”也不表示 allocator 已经空闲，必须先走淘汰和释放的所有权转移。

<a id="radix-tree"></a>
## 5. KV page 与 radix cache 所有权

```mermaid
flowchart TD
    Free[free page] -->|allocator alloc| ReqOwned[active request page]
    ReqOwned -->|committed full page insert| Cached[radix-owned page]
    Cached -->|match + inc_lock_ref| Protected[protected cache page]
    Protected -->|dec_lock_ref| Cached
    Cached -->|LRU evict| Free
    ReqOwned -->|finish/retract/rollback| Free
```

这张图里的“所有权”不是说 KV tensor 被复制或 page 有两个 allocator owner。一个 page
始终只有两种 allocator 状态：`free` 或 `allocated`。图中所谓“request-owned”与
“radix-owned”说的是：**谁决定这批已分配 slots 接下来能否释放、能否给其他请求复用**。
请求命中 cache 后，request row 仍会指向同一批 slot；它只是持有 cache 的 lock/ref，不能
自行释放它们。

| 图中状态 | allocator 看见的 page 状态 | 谁持有/管理 slot 的生命周期 | 请求 row 是否可指向它 | 能否立刻给新请求分配 | 下一步可能发生什么 |
|---|---|---|---|---|---|
| `free page` | `free` | 没有持有者 | 否 | 能，allocator 可分配整页 | 分配给某个请求；page 0 仅作 padding，不会走这条路径。 |
| `active request page` | `allocated` | 活跃请求的私有 KV；包括未满页尾部和尚未 committed 的预分配范围 | 是，`req_to_token[row, pos] -> slot` | 不能；即便尾 slot 尚未写入，也只能由同一请求续写 | scheduler commit 后变为 committed（同步路径在 runner 返回后，overlap 路径在 launch 成功后）；完整 page 可插入 cache，失败/rollback/retract/结束时可释放其非共享 page。 |
| `radix-owned page` | `allocated` | `MiniRadixCache` 保存 token-prefix → slot 的可复用映射，决定其是否可被 LRU 淘汰 | 可以；旧请求或新命中请求都可指向同一 slot | 不能；它仍是已分配 page | 新请求命中时加 lock/ref；没有 lock 时可被 LRU 淘汰。 |
| `protected cache page` | `allocated` | radix cache 仍是生命周期 owner；一个或多个活跃请求只是借用者 | 是 | 不能 | 所有借用请求结束/换节点后 `dec_lock_ref`；ref 归零后回到可驱逐 cache。 |
| `evictable cache page` | `allocated` | radix cache，且没有活跃请求借用 | 否（没有活跃请求），但未来请求仍可 prefix-hit | 不能直接分配；它只是“可回收预算” | LRU 选中后从 radix tree 删除，scheduler 再调用 allocator `free()`，才真正回到 `free page`。 |

这里的两个容易误解之处是：

- `protected` 不是一种新的物理 page，也不是 request 把 page 从 cache 手里拿走；它是
  `radix-owned page + lock_ref > 0`。
- `evictable_size()` 也不是 allocator 的 `available_size()`。前者必须先经过“从树中移除
  → 交给 allocator 释放”的所有权转移，才能成为后者。

### 例子：A 写入 prefix，B 命中并复用同一页

设 `page_size=2`。page 0 的 `[0,1]` 是 padding；可分配的 page 1、2、3 分别是
`[2,3]`、`[4,5]`、`[6,7]`。请求 A 先处理 token `[11,12,13]`：

| 时刻 | A 的逻辑位置 → slot | page 归属 / lock | 为什么 |
|---|---|---|---|
| A 刚完成 `[11,12]` | `0→2, 1→3` | page 1 已分配给 A | `[11,12]` 已 committed，且刚好占满一个 page。 |
| A 将完整 prefix 插入 radix tree 并 pin | 不变，仍是 `0→2, 1→3` | page 1：radix-owned + protected，`lock_ref=1` | tree 记录 `[11,12] → [2,3]`；A 的 row 继续引用这些 slot，但无权自行 free page 1。 |
| A 再处理 `13` | `2→4` | page 2：A 私有；page 1 状态不变 | page 2 的 slot 5 是尾部空位，只能由 A 续写，不能给别的请求使用。 |

此时请求 B 到达，prompt 是 `[11,12,99]`。它对 radix tree 的匹配和 suffix 分配如下：

| B 的阶段 | B 的逻辑位置 → slot | page 归属 / lock | 发生了什么 |
|---|---|---|---|
| `match_prefix([11,12,99])` | `0→2, 1→3` | page 1：radix-owned + protected，`lock_ref=2` | B 复用 A 已经算好的 KV；不会重新为 `[11,12]` 分配 page。 |
| 为 suffix `99` 分配 slot | `2→6` | page 3：B 私有 | B 不能使用 A 的 page 2 尾 slot 5，因此拿一个自己的整页；slot 7 留给 B 后续续写。 |

接下来按时间释放：

1. A 结束：A 对 page 1 的 lock 减一，`lock_ref=1`，因为 B 仍在使用，page 1 不能淘汰；
   A 私有的 page 2 没有 cache 保护，所以 allocator 可将其释放。
2. B 结束：B 的 lock 也减一，page 1 的 `lock_ref=0`，它变为 `evictable cache page`；B
   的不满 page 3 不是完整 cache prefix，直接释放。
3. 内存有压力时：LRU 从 radix tree 移除 `[11,12] → [2,3]`，并把这些 slots 交给
   allocator `free()`；只有这一步完成后，page 1 才重新出现在 `available_size()` 中。

因此，“请求结束”不等于“它曾经使用过的每个 page 都立刻 free”：完整、已缓存的 prefix
会留下来服务未来请求；私有 suffix、未满页尾部、rollback 的 overallocated 范围才随请求释放。

[`MiniRadixCache.match_prefix()`](../src/my_sglang/radix_cache.py) 和 [`insert()`](../src/my_sglang/radix_cache.py) 都向下截断到完整 page。锁住命中终点时，[`inc_lock_ref()`](../src/my_sglang/radix_cache.py) 会沿父链增加引用；释放时必须从同一终点 [`dec_lock_ref()`](../src/my_sglang/radix_cache.py)。

- `protected_size()`：活跃请求正在借用，不能淘汰。
- `evictable_size()`：没有 lock，可计入 admission 可回收预算。
<a id="radix-lru"></a>

- [`evict(n)`](../src/my_sglang/radix_cache.py)：按 LRU 删除未锁定叶子，返回 slot；真正 free page 的仍是 scheduler。

page 是释放粒度。`free_unshared_pages(candidate, protected)` 只释放与 `protected` 不共页的 candidate，防止 cache prefix 与请求尾部共享一页时误释放。

## 6. Admission 的 `MemoryBudget`

[`PrefillAdder`](../src/my_sglang/schedule_policy.py) 在一次规划中维护：

```text
remaining_tokens
  = allocator.available_size
  + cache.evictable_size
  - running decode reserve

remaining_prefill_tokens = max_prefill_tokens
```

每个候选成本还包含 page 向上取整、输出预留和一页对齐余量。返回值语义：

| 结果 | 状态变化 |
|---|---|
| `ADMIT` | 完整 suffix 进入本轮 EXTEND |
| `CHUNK` | 只推进一段，成为唯一 `chunked_req` |
| `DEFER` | 保留 waiting；FCFS 不越过首个预算失败请求 |
| `ABORT` | 物理上连最小执行块都无法容纳，结束并给出原因 |

空系统首请求或正在继续的 chunk 有“物理可容纳”兜底，避免因为未来输出预留永久 defer；真正的后续压力由 decode evict/retract/abort 闭环处理。

## 7. 一组可以随时断言的不变量

[`MiniScheduler.assert_consistent()`](../src/my_sglang/scheduler.py) 汇总检查：

```text
request rows: active + free == size，且集合不相交
KV pages: allocated + free == num_pages，page 0 不在两者中
每个活跃请求的已映射 slot 都属于 allocated page
committed 不领先 allocated
cache protected 不领先 committed
pipeline overlap 普通路径 launch 后 allocated == committed，CPU output/result 可滞后一批
result_queue 深度 <= 2，`_inflight_refs` 必须等于队列中按请求统计的引用数
逻辑 FINISHED 但尚有 in-flight owner 时允许暂缓 row/KV/runner 释放
```

建议调试时同时打印 [`memory_snapshot()`](../src/my_sglang/scheduler.py)：free、allocated、mapped、cache evictable/protected 和 decode reserve 能快速说明“内存去哪了”。
