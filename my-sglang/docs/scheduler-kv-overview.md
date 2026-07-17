# Scheduler / KVCache 核心结构

这是一份先读文档：先用六张图建立全局结构，再去看字段和逐行流程。`my-sglang` 只追踪 CPU 上的索引和所有权，不保存每层真实 K/V 张量，但调度名词和方法边界尽量贴近 SGLang。

## 1. 调度主循环

```mermaid
flowchart TD
    Settle["_settle_last_batch<br/>filter / merge"] --> Prefill{"PrefillAdder<br/>有 ADMIT/CHUNK?"}
    Prefill -->|yes| Extend["MiniScheduleBatch<br/>EXTEND"]
    Prefill -->|no| DecodeMem{decode page 足够?}
    DecodeMem -->|no| Pressure["cache evict<br/>→ retract<br/>→ abort"]
    Pressure --> DecodeMem
    DecodeMem -->|yes| Decode["MiniScheduleBatch<br/>DECODE"]
    Extend --> Forward[runner forward]
    Decode --> Forward
    Forward --> Commit["commit / finish / chunk cache<br/>save as last_batch"]
    Commit --> Settle
```

[`MiniScheduler.step()`](../src/my_sglang/scheduler.py#L117) 一次只执行 EXTEND 或 DECODE；[`_settle_last_batch()`](../src/my_sglang/scheduler.py#L183) 在下一轮把上轮存活请求合入 `running_batch`。测试入口：[`test_prefill_priority_means_one_forward_batch_per_step`](../tests/test_scheduler.py#L102)。

### 1.1 与标准 SGLang 的主循环差异

上图是**教学主线**，不是标准 SGLang 所有可选路径。它故意固定为“Prefill 优先；本轮只跑 EXTEND 或 DECODE”，这样 slot、page 与请求状态的因果关系最清楚。

| 维度 | `my-sglang` | 标准 SGLang | 影响 |
|---|---|---|---|
| 本轮 batch 模式 | 只会是 `EXTEND` 或 `DECODE` | 除 EXTEND / DECODE 外，还可通过 `mix_with_running()` 形成 `MIXED` batch | 真实一轮可同时带新 prefill 与已有请求的 decode 部分 |
| waiting 选择 | 固定 FCFS；第一个 `DEFER` 阻挡后续请求 | 可选 FCFS、LPM、DFS-weight、LOF、Random、Routing-key，并可叠加 priority | “谁先 admission”不一定按到达顺序 |
| 本轮外围工作 | 只做 KV 调度和 runner 调用 | 还处理超时/取消、LoRA、分布式与 disaggregation、采样等 | 这些不改变 slot 主线，却会改变实际可调度请求集合 |

```text
教学版：  waiting ──FCFS──> EXTEND 或 DECODE（二选一）
真实版：  waiting ──策略排序──> EXTEND / DECODE / MIXED
                              └─ 还受模型、缓存层、并行与运行时条件约束
```

真实代码证据：[`SchedulePolicy` 支持的策略](../../python/sglang/srt/managers/schedule_policy.py#L101)、[`mix_with_running()`](../../python/sglang/srt/managers/schedule_batch.py#L2377)、[`get_next_batch_to_run()`](../../python/sglang/srt/managers/scheduler.py#L2555)。后文第 4 节的 FCFS 预算叙述均特指教学版；第 6 节另说明 overlap 差异。

## 2. `MiniScheduleBatch` 生命周期

```mermaid
stateDiagram-v2
    [*] --> Scheduled: 建 batch(reqs + mode)
    Scheduled --> Prepared: prepare_for_extend/decode；申请 slot，写 row
    Prepared --> Pending: to_forward_batch + start/kick（仅 overlap）
    Prepared --> Forwarded: runner 同步返回（普通 step）
    Pending --> Forwarded: finalize 成功
    Forwarded --> Committed: commit_allocated；allocated = committed
    Forwarded --> RolledBack: runner / finalize 失败
    Committed --> LastBatch: 写 token / 更新请求状态
    LastBatch --> RunningBatch: 下一轮 settle（EXTEND merge；DECODE 交棒）
    RolledBack --> Waiting: rollback + release + reset
```

先给结论：`MiniScheduleBatch` 是**一次 forward 的可变工作单**，不是请求本身，也不是长期的 KV 容器。它收集本轮要跑的请求，申请本轮新增 KV slot，生成给 runner 的参数，并在成功/失败后推进或回滚这次申请。

```text
Req（跨多轮长期存在）
  rid / output_ids / req_pool_idx / KV 进度
          │ 被选中一次
          v
MiniScheduleBatch（只代表本轮 forward）
  reqs + mode + 本轮 input / 新 slot / 长度元数据
          │ 转为 tuple 化参数
          v
BatchForward（交给 runner 的结构化视图）
  不再负责分配、提交或回滚
```

### 2.1 谁创建、谁持有、何时消失

先记这一句：**`batch` 不是第三个长期容器，只是本轮的局部变量名；它会指向新对象，或直接指向 `running_batch`。** `last_batch` 与 `running_batch` 是 Scheduler 的两个字段；某些时刻会引用同一个对象，但用途不同。

```text
名称              谁持有                  它回答的问题
batch             当前 step 的局部变量     “本轮要 forward 的是哪一批？”
last_batch        Scheduler 字段           “上一轮完成、下一轮还要 settle 的是哪一批？”
running_batch     Scheduler 字段           “下一次 decode 的候选请求集合是什么？”
```

图中只用两个对象代号：`P` = 本轮新建的 Prefill batch；`D` = 原来就在 `running_batch` 中、这轮复用来 Decode 的 batch。箭头 `→` 表示“字段/变量引用这个对象”，**不是复制对象**。

#### 2.1.1 EXTEND：新对象 P 的请求被合并进旧运行集合 R

```mermaid
flowchart LR
    E0["Step N 开始<br/>running_batch → R<br/>last_batch = None"]
    E1["选中 Prefill<br/>新建 P<br/>local batch → P<br/>running_batch 仍 → R"]
    E2["P forward 成功<br/>last_batch → P<br/>running_batch 仍 → R"]
    E3["Step N+1 settle<br/>过滤 P 中 finished / chunked<br/>R.merge_batch(P)<br/>last_batch = None"]
    E4["结果<br/>running_batch 仍 → R<br/>P 的存活请求已加入 R"]
    E0 --> E1 --> E2 --> E3 --> E4
```

这条路径的重点是：**`running_batch` 不会指向 P。** 它一直是原对象 R；settle 只是把 P 中仍要 decode 的请求追加到 R 的 `reqs` 列表。因此 `last_batch = None` 后，P 不再由 Scheduler 字段持有，但请求本身已经在 R 中继续存活。

#### 2.1.2 DECODE：同一个对象 D 在 `last_batch` 与 `running_batch` 间交棒

```mermaid
flowchart LR
    D0["Step N 开始<br/>running_batch → D<br/>last_batch = None"]
    D1["选中 Decode<br/>local batch → D<br/>没有新建 Decode batch"]
    D2["D forward 成功<br/>last_batch → D<br/>running_batch → 新空对象 R0"]
    D3["Step N+1 settle<br/>过滤 D 中 finished 请求<br/>running_batch → D<br/>last_batch = None"]
    D4["结果<br/>D 成为下一轮 Decode 候选集合"]
    D0 --> D1 --> D2 --> D3 --> D4
```

这条路径的重点是：**D 没有被复制。** `batch`、随后 `last_batch`、最后 `running_batch` 都依次引用同一个 D。中间先放入空 `R0`，是为了保证“刚跑完的 D”必须等到下一轮 settle 后才能再次作为 decode 候选集合。

#### 2.1.3 Overlap：P / D 先由 `_pending` 暂存，再进入 `last_batch`

```mermaid
flowchart LR
    O0["launch_step<br/>local batch → P 或 D<br/>_pending.batch → 同一对象<br/>last_batch = None"]
    O1["pending 窗口<br/>allocated 已推进<br/>committed 尚未推进"]
    O2["finalize_pending 成功<br/>处理 token<br/>last_batch → P 或 D<br/>_pending = None"]
    O3["下一次 launch_step<br/>调用 settle<br/>按 EXTEND 合并或按 DECODE 交棒"]
    O0 --> O1 --> O2 --> O3
```

Overlap 与普通 `step()` 的差别只在于：普通版在一个调用里完成 forward 与 `last_batch = batch`；Overlap 先让 `_pending.batch` 持有对象，等 finalize 成功后才交给 `last_batch`。

| 时刻 | `batch`（局部变量） | `last_batch` | `running_batch` | 是否同一对象 |
|---|---|---|---|---|
| Scheduler 初始化 | 无 | `None` | 空对象 `R0` | — |
| 本轮选中 EXTEND | 新对象 `Bp` | `None` | 原运行集合 `R` | `Bp ≠ R` |
| EXTEND 成功后 | `Bp` | `Bp` | 仍是 `R` | `batch` 与 `last_batch` 同一个；`running_batch` 不变 |
| 下轮 settle EXTEND | 临时读 `Bp` | 清为 `None` | 原 `R` 执行 `merge_batch(Bp)` | `R` 不会变成 `Bp`；只是把 Bp 中存活请求加入 R |
| 本轮选中 DECODE | `Rd` | `None` | `Rd` | **`batch is running_batch`** |
| DECODE 成功后 | `Rd` | `Rd` | 新空对象 `R0` | `batch` 与 `last_batch` 同一个；旧 `Rd` 已不再叫 `running_batch` |
| 下轮 settle DECODE | 临时读 `Rd` | 清为 `None` | 直接设回 `Rd` | **对象交棒：`running_batch = last_batch`** |

`chunked_req` 是 EXTEND 的例外：chunk 未完成时，`_settle_last_batch()` 会从 `Bp` 中排除它；它留在独立的 `chunked_req` 字段，下一轮继续 prefill，不会先合进 `running_batch`。

所谓 batch “消失”不是显式 `free`：当 `_settle_last_batch()` 把 `last_batch = None`，且局部 `batch` 离开 `step()` 作用域后，Scheduler 不再持有这个 batch 对象；其中存活的 `Req` 已被合并或交棒到 `running_batch`，finished 请求则已释放 row / KV。

Overlap 还多一个临时持有者：`launch_step()` 后，batch 由 `_pending.batch` 持有；在 `finalize_pending()` 成功前不会写入 `last_batch`，失败则走 rollback。这是第 6 节的两阶段边界，不应和普通 `last_batch` 混在一起。

源码入口：[`_get_new_prefill_batch()`](../src/my_sglang/scheduler.py#L199)、[`_get_decode_batch()`](../src/my_sglang/scheduler.py#L276)、[`_settle_last_batch()`](../src/my_sglang/scheduler.py#L183)、[`launch_step()` / `finalize_pending()`](../src/my_sglang/overlap_scheduler.py#L90)。

### 2.2 每一步实际改了什么

| 时刻 | `Req` 上的变化 | `ReqToTokenPool[row, seq_pos]` | Batch 内字段 | 能否运行模型 |
|---|---|---|---|---|
| 建 batch 前 | 新请求已分到 `req_pool_idx`；命中 cache 的 prefix 已写入 | 只有 cache prefix（或空） | 只有 `reqs`、`forward_mode` | 否 |
| `prepare_for_extend()` | `kv_allocated_len: start → fill_len` | 写入新 suffix 的 slot | `input_ids_by_req`、`out_cache_locs_by_req`、`seq_lens`、`prefix_lens` | 可以 |
| `prepare_for_decode()` | 每个请求 `kv_allocated_len + 1` | 每行写一个新 slot | 每个请求一个 `last_token_id` 和一个 slot | 可以 |
| `to_forward_batch()` | 不改 `Req` | 不改表 | NumPy 数组 / list 转成 tuple 视图 | 可以 |
| runner 成功 + `commit_allocated()` | `kv_committed_len = kv_allocated_len` | 这些 slot 正式成为请求已完成 KV | 元数据不变 | 已完成 |
| runner 失败 + `rollback_uncommitted()` | `kv_allocated_len` 回到 `kv_committed_len` | 清掉未提交区间；只释放不与已提交尾页共用的 page | 本轮 allocation 作废 | 不可继续 |

> **边界纠正**：`MiniScheduleBatch` 负责本轮**新增** slot 的申请和写入；它不是所有 mapping 的唯一写入者。新请求命中 radix cache 时，调度器的 [`_attach_new_request()`](../src/my_sglang/scheduler.py#L367) 会先把已命中的 `prefix_indices` 写到 row；随后 batch 只补上未命中的 suffix。

### 2.3 `allocated` 与 `committed`：用一轮数字看清

假设请求 A 已命中 4 个 cache token，接下来 prefill 3 个新 token；其 row 中已有 `slot 4～7`。

```text
时刻                         KV 进度                         row 0（只画 0~6）
建 batch 前                  allocated = committed = 4       [4, 5, 6, 7, -1, -1, -1]
prepare_for_extend() 后      allocated = 7, committed = 4     [4, 5, 6, 7, 12, 13, 14]
runner 成功 + commit 后      allocated = committed = 7       [4, 5, 6, 7, 12, 13, 14]
runner 失败 + rollback 后    allocated = committed = 4       [4, 5, 6, 7, -1, -1, -1]
```

这里的 `allocated` 是“为本轮 forward 预先占住的**物理 KV slot** 数”；`committed` 是“模型已成功处理、可以作为下轮前缀使用的 KV slot 数”。二者相等时，没有悬而未决的分配。Overlap 的 `launch_step()` 之后会短暂看到 `allocated > committed`；`finalize_pending()` 成功才追平，失败则回滚。这正是两个字段同时存在的原因。

| 模式 | `prepare` 后 | runner 完成后 | 观察点 |
|---|---|---|---|
| 普通 `step()` | 短暂 `allocated > committed`，外部通常看不到 | 同一调用内立即 commit | 返回 `StepResult` 时通常已相等 |
| Overlap `launch_step()` | `allocated > committed` | 尚未 finalize，不能当作已完成 KV | `_pending` 存在 |
| Overlap `finalize_pending()` | 不再新增 allocation | 成功：二者相等；失败：回到旧 committed 值 | `_pending = None` |

对应测试直接断言了这条时间线：[`test_launch_allocates_and_finalize_commits_prefill_and_decode`](../tests/test_overlap_scheduler.py#L90)。

### 2.4 EXTEND、chunk 与 DECODE 如何走同一套生命周期

| 本轮类型 | `input_ids_by_req` | 新增 slot 数 | forward 后的请求去向 |
|---|---|---:|---|
| 首次 EXTEND / prefill | prompt 中未命中 cache 的部分 | `fill_len - kv_allocated_len` | 最后一个 chunk：生成首个 output，转 `RUNNING`；否则转 `PREFILLING` |
| 后续 chunk EXTEND | 下一个 prompt 分块 | 当前 chunk 长度 | 已完成完整 page 可写入 radix cache；未完成 chunk 留在 `chunked_req` |
| DECODE | 上一个生成 token（`last_token_id`） | 每个请求固定 1 | 追加新 output；结束则释放 row / KV，否则下轮继续 |

```mermaid
flowchart LR
    A["Req A: prefix 已 committed"] --> B["EXTEND batch\nprepare: 分配 suffix slot"]
    B --> C{"最后一个 prefill chunk?"}
    C -->|否| D["commit → PREFILLING\n缓存完整 page → chunked_req"]
    D --> B
    C -->|是| E["commit → append 首 token\nRUNNING"]
    E --> F["DECODE batch\nprepare: 每 req +1 slot"]
    F --> G{"EOS / 长度到达?"}
    G -->|否| F
    G -->|是| H["FINISHED\n释放 row 与非缓存 KV"]
```

### 2.5 `BatchForward` 是什么，为什么还要再造一个对象

`MiniScheduleBatch` 有 allocator、pool、cache 等**可变控制面**；runner 只需要本轮输入、slot 和长度。因此 [`to_forward_batch()`](../src/my_sglang/schedule_batch.py#L167) 把它缩成 `BatchForward`：

| `BatchForward` 字段 | runner 用它做什么 |
|---|---|
| `mode`、`reqs` | 知道是 EXTEND 还是 DECODE，以及对应请求顺序 |
| `input_ids_by_req` | 本轮实际送入模型的 token；不是整个历史 prompt |
| `req_pool_indices`、`out_cache_locs` | 将本轮 token 写到正确的 KV 逻辑行 / 物理 slot |
| `seq_lens`、`extend_lens`、`chunk_starts_by_req` | 识别本轮后 token 边界和 chunk 位置 |
| `prefix_slot_ids_by_req` | 首次 prefill 命中 cache 时，告诉 runner 可复用的 KV 前缀 |

它是 `@dataclass(frozen=True)`，且数组被转为 tuple，防止 runner 改动 batch 的容器参数；但 `reqs` 内仍是同一批 `Req` 对象的引用，**不是对请求状态的深拷贝**。因此真正的 allocation、commit、rollback 仍必须回到 `MiniScheduleBatch` / scheduler 执行，不能由 runner 直接修改。

核心方法：[`prepare_for_extend()`](../src/my_sglang/schedule_batch.py#L60)、[`prepare_for_decode()`](../src/my_sglang/schedule_batch.py#L99)、[`commit_allocated()`](../src/my_sglang/schedule_batch.py#L129)、[`rollback_uncommitted()`](../src/my_sglang/schedule_batch.py#L136)、[`filter_batch()` / `merge_batch()`](../src/my_sglang/schedule_batch.py#L155)。

## 3. KV 索引与物理 page

先记住这条主线：**token 的序号是逻辑位置；slot 是 KV 的索引；page 是一次申请、释放和复用的最小块。** 这里的 `my-sglang` 只模拟索引和归属，真实模型的 K/V 张量不在这个教学实现里。

| 名词 | 它是什么 | 例子（`page_size = 4`） |
|---|---|---|
| `seq_pos` | token 在一个请求序列中的逻辑下标 | 第 0、1、2 个 token |
| `slot` | 该 token 的 KV 存放位置索引 | `4`、`5`、`6` |
| `page` | 连续 `page_size` 个 slot 的分配块 | page 1 = slot 4～7 |
| `req_pool_idx` | 请求在索引矩阵中的行号 | `req-A → row 0` |

### 3.1 KV Slot 到底是什么

**Slot 就是“某个 token 的 KV 缓存格号”。** 把 KV cache 想成一排储物柜：token 不直接拿着 K/V 张量，而是拿着一个柜号；attention backend 用这个整数索引读写该 token 对应的 K/V。这个项目中的 slot 是 `int64` 索引，**不是 token id、不是 GPU 指针、也不是一个 page**。

```text
请求 A 的第 2 个 token
         |
         |  req_to_token[row_A, 2] = 6
         v
      slot 6  ─────────────→ 真实 KV cache 中该 token 的 K/V 位置
      （柜号）                  （本教学项目不保存张量）

slot 6 所在 page：page_id = 6 // 4 = 1
slot 6 在页内的位置：page_offset = 6 % 4 = 2
```

| 容易混淆的对象 | 与 Slot 的关系 |
|---|---|
| token id | token 的内容编号；相同 token id 在两个请求中可落到不同 slot |
| `seq_pos` | token 在**本请求**中的逻辑位置；用它到行表里查 slot |
| page | 一组连续 slot；`page_size = 4` 时，一个 page 有 4 个 slot |
| 真实 K/V 张量 | slot 是访问它们的索引；`my-sglang` 不保存张量本身 |

Slot 的价值是把“请求里的第几个 token”和“KV cache 的实际占位”解耦：同一段新分配后缀通常顺序填 slot；命中 radix 前缀时，新请求可直接复用旧 slot，所以一个请求跨过“缓存前缀 → 新后缀”的边界时，slot 也可能跳号。

源码中，allocator 先产出 slot 列表，`prepare_for_extend()` / `prepare_for_decode()` 再把它写入 `req_to_token[row, seq_pos]`。见 [`alloc_extend()`](../src/my_sglang/pools.py#L149)、[`ReqToTokenPool.write()`](../src/my_sglang/pools.py#L52)、[`prepare_for_extend()`](../src/my_sglang/schedule_batch.py#L60)。

### 3.2 一个 page 长什么样

下面只是在画**索引编号**，不是实际 GPU 内存地址。`slot = page_id × page_size + page_offset`；page 0 留给 padding，分配器从 page 1 开始分配。

```text
page_size = 4

                 page 0（保留，不能分配）
              +------+------+------+------+
slot 索引      |  0   |  1   |  2   |  3   |
              +------+------+------+------+

                 page 1（一次整体申请）
              +------+------+------+------+
slot 索引      |  4   |  5   |  6   |  7   |
page_offset    |  0   |  1   |  2   |  3   |
              +------+------+------+------+

                 page 2（下一块）
              +------+------+------+------+
slot 索引      |  8   |  9   | 10   | 11   |
              +------+------+------+------+
```

### 3.3 请求的 KV 索引怎样落到 page

> **实现边界**：本节图中的 `NumPy int64[row, seq_pos]` 是 `my-sglang` 的教学实现，运行在 CPU。真实 SGLang 保留同一个 `(req_pool_idx, seq_pos) → slot` 关系，但 `ReqToTokenPool.req_to_token` 是运行设备上的 `torch.int32` tensor；CUDA 服务时它就在 GPU 上，供 attention / KV cache 路径直接索引。真实实现还保留 row 0 给 CUDA Graph 的 padding，真实请求从 row 1 起分配。见 [真实 SGLang `ReqToTokenPool`](../../python/sglang/srt/mem_cache/memory_pool.py#L229)。

`ReqToTokenPool` 的形状固定为 `[max_running_reqs, max_context_len]`。所以 **row 0 不会只有 4 格**：它有完整的 `max_context_len` 列；图只截取 16 格，未映射位置是 `-1`。调度时只关心 `[0, kv_allocated_len)` 这个已分配前缀。

| 参数 / 字段 | 管什么 | 与另一个概念的关系 |
|---|---|---|
| `max_running_reqs` | 矩阵有多少**行**，也就是同时最多有多少请求占用行 | 不是 reqId 的数量上限；请求结束，行可被下一个 reqId 复用 |
| `rid`（reqId） | 请求的业务身份，例如 `req-A` | `_rid_to_idx` 把它映射到暂时的 `req_pool_idx` |
| `req_pool_idx` | 一个活跃请求当前占用的行号 | 只是 0～`max_running_reqs - 1` 的临时槽位，不是请求 ID |
| `max_context_len` | 每行有多少**列**，也是单请求最大可映射 token 数 | `seq_pos` 必须在 `0～max_context_len - 1` 内 |
| `seq_pos` | token 在本请求内的第几个逻辑位置 | 它是列下标，不是 token id，也不是 slot |

```text
例：max_running_reqs = 3，max_context_len = 8

活跃请求身份表（请求结束后，row 可复用）       KV 索引矩阵（每个 row 固定 8 列）
rid       req_pool_idx                         seq_pos →  0   1   2   3   4   5   6   7
req-A  →  row 0                              row 0 / A | 4 | 5 | 6 | -1| -1| -1| -1| -1|
req-B  →  row 1                              row 1 / B |12 |13 | -1| -1| -1| -1| -1| -1|
（空闲）   row 2                              row 2     |-1 |-1 |-1 |-1 |-1 |-1 |-1 |-1|

req-A 结束：row 0 清空；新来的 req-C 可以占用 row 0。
req-A 的 seq_pos = 2：表示 A 的第 3 个 token，不表示全局第 3 个 token。
```

#### `my-sglang` 的 NumPy 矩阵在简化什么代码？

`req_to_token` 的本质只是“**请求逻辑位置 → KV slot**”的二维查找表；它不保存真实 K/V 张量，也没有引入新的调度规则。这里用 NumPy 是教学版把原本需要逐格循环的整数表操作，写成固定形状的整段操作。

| 实现 | 索引表类型 | 放在哪里 | 目的 |
|---|---|---|---|
| `my-sglang` | `np.ndarray[int64]` | CPU | 代码短、便于观察矩阵和切片 |
| 真实 SGLang | `torch.Tensor[int32]` | `device`（CUDA 时为 GPU） | 让 GPU 侧 attention / KV cache 路径直接使用索引；row 0 作为 CUDA Graph padding |

| 要做的事 | 不用 NumPy 时的直觉写法 | 这里的 NumPy 写法 |
|---|---|---|
| 初始化 / 释放一行 | 循环把每个格子设为 `-1` | `row.fill(-1)` |
| 为一段新 token 写 slot | 循环写 `start` 到 `end - 1` | `matrix[row, start:end] = slots` |
| 算 batch 中每个请求本轮新增多少 token | 对每个请求做 `target - start` | `seq_lens - prefix_lens` |
| 数有多少格已映射 | 双层循环数“不是 `-1`”的格子 | `np.count_nonzero(matrix >= 0)` |

```text
例：同一 batch 有 req-A、req-B

                 req-A  req-B
prefix_lens        5      0      # 本轮之前已有 / 命中的 KV 长度
seq_lens           8      2      # 本轮结束后应有的 KV 长度
--------------------------------
extend_lens        3      2      # 本轮各自要新增的 token 数

NumPy 的逐项相减： [8, 2] - [5, 0] = [3, 2]
等价 Python 循环：   [8 - 5, 2 - 0] = [3, 2]
```

`np.count_nonzero` 也不是在数“slot 数值本身有多大”。代码先计算 `matrix >= 0`，把已映射位置变成 `True`、`-1` 变成 `False`，再数 `True` 的个数：

```text
slot 表                 matrix >= 0                 已映射数
[[ 4,  5, -1],          [[ T,  T, F],
 [12, -1, -1]]    →      [ T,  F, F]]        →          3
```

因此，NumPy 在这个教学实现里是**CPU 侧索引代码的简化工具**：固定 `int64` 类型、连续切片和批量运算更好写；真正的 K/V 张量不在这里，送给 runner 前的快照也会转回普通 Python 整数。不要把它理解为标准 SGLang 的物理实现：标准 SGLang 用设备上的 `torch.int32` 索引 tensor，索引关系相同，但能直接服务 GPU kernel。

在 `MiniScheduler` 中，`ReqToTokenPool(max_running_reqs, max_context_len or max_total_tokens)` 建立这张表；写入超出 `max_context_len` 会直接报错。见 [`MiniScheduler.__init__()`](../src/my_sglang/scheduler.py#L44)、[`ReqToTokenPool.alloc()`](../src/my_sglang/pools.py#L26)、[`ReqToTokenPool.write()`](../src/my_sglang/pools.py#L54)。

下面用一个更接近实际的例子：`max_context_len = 32`、`page_size = 4`。`req-A` 命中 8-token radix 前缀，随后 prefill 6 个新 token，再 decode 2 个 token。假设其他运行请求已占用 page 3～6，allocator 当前可用队首是 page 7、8；因此 A 的逻辑 token 连续，slot 却在位置 8 跳号。

```text
请求 req-A  ── req_pool_idx = 0
row 0 的真实宽度 = max_context_len = 32；这里只画 seq_pos 0～15

ReqToTokenPool（逻辑位置连续；slot 可以跳号）
seq_pos       0     1     2     3  |  4     5     6     7  |  8    9    10   11 | 12   13 | 14   15 | 16...
             +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-------
row 0 / A    |  4  |  5  |  6  |  7  |  8  |  9  | 10  | 11  | 28  | 29  | 30  | 31  | 32  | 33  | 34  | 35  | -1...
             +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-------
来源          \____________ radix cache 前缀 ____________/ \________ 新分配 ________/ \_尾页复用_/  未映射
page          page 1（slot 4～7）    page 2（slot 8～11）    page 7（28～31）   page 8（32～35）

其他运行请求暂时占用 page 3～6；所以 req-A 不会拿到 slot 12～27。
后续 seq_pos 16 首次写入时，才会再申请一个新 page。
```

| 位置范围 | slot | 含义 |
|---|---|---|
| `seq_pos 0～7` | `4～11` | radix 命中的完整 page 前缀，不重新计算 K/V |
| `seq_pos 8～13` | `28～33` | 这次 prefill 新分配的 6 个位置 |
| `seq_pos 14～15` | `34～35` | decode 继续填 page 8 的尾页，不申请新 page |
| `seq_pos 16～31` | `-1` | 还未映射；请求结束后整行会清回 `-1` |

```mermaid
flowchart LR
    R["请求 req-A"] --> I["req_pool_idx = 0"]
    I --> M["req_to_token[0, seq_pos]"]
    M -->|"seq_pos 0~3 → slot 4~7"| P1["cache page 1"]
    M -->|"seq_pos 4~7 → slot 8~11"| P2["cache page 2"]
    M -->|"seq_pos 8~11 → slot 28~31"| P7["new page 7"]
    M -->|"seq_pos 12~15 → slot 32~35"| P8["new page 8 / 尾页复用"]
    P3["page 3~6<br/>被其他请求占用"] -. "不能给 A" .-> P7
    P1 --> K["真实 K/V 张量的位置<br/>此教学实现不保存"]
    P2 --> K
    P7 --> K
    P8 --> K
```

### 3.4 为什么 allocator 按 page 管，而不是按 token 管

| 情况 | slot 表中的变化 | page 的变化 |
|---|---|---|
| 命中 8-token radix 前缀 | 写入已存在的 slot `4～11` | **不申请 page**，直接复用 cache |
| prefill 6 个新 token | 写入 slot `28～33` | 申请 page 7、8；page 8 还剩 2 格 |
| 后续 decode 2 token | 写入 slot `34、35` | **不申请新 page**，填满 page 8 尾页 |
| 再 decode 1 token | 写入下一个 page 的首 slot | 申请一个新 page |
| 回滚失败的 forward | 清掉未提交的索引格 | 只归还没有被 committed 位置共用的 page |
| radix cache 缓存前缀 | 新请求的前缀格可直接指向旧 slot | 只缓存完整 page；未满尾页仍归请求 |

所以，`page_size` 越大，分配/回收次数越少，但序列尾部可能暂时占着未写满的一小段空间；`page_size = 1` 时就退化为“一 token 一 page”。

```mermaid
flowchart LR
    Req["Req<br/>req_pool_idx<br/>allocated/committed"] --> Matrix["ReqToTokenPool<br/>my-sglang: NumPy int64[row, seq_pos]"]
    Matrix --> Slot["slot<br/>一个 token 的 KV 索引"]
    Slot --> Page["Token/Paged allocator<br/>按 page 申请/释放；page 0 保留"]
    Radix["MiniRadixCache<br/>完整 page 前缀：evictable/protected"] --> Slot
    Req -. last_node lock .-> Radix
```

唯一映射事实是 `req_to_token[row, seq_pos] = slot`。allocator 以 page 为分配/释放单位，并允许同一序列继续使用已分配尾页；radix cache 只接管 page 对齐的完整前缀。代码入口：[`ReqToTokenPool`](../src/my_sglang/pools.py#L12)、[`BaseTokenToKVPoolAllocator`](../src/my_sglang/pools.py#L101)、[`MiniRadixCache`](../src/my_sglang/radix_cache.py#L100)；测试入口：[`test_paged_allocator_reuses_tail_before_allocating_next_page`](../tests/test_pools.py#L41)。

### 3.5 `MiniRadixCache` 与真实层级缓存的范围

本节前面的 slot / page 映射关系可以直接迁移到真实 SGLang；但 **cache 的存储层级不能直接迁移**。`MiniRadixCache` 只模拟本地 radix tree 的 prefix ownership、lock 和 LRU，不保存真实 K/V，也没有异步搬运。

```text
my-sglang
请求 ──> MiniRadixCache（本地 prefix → slot）──> 教学 allocator

标准 SGLang（按配置启用）
请求 ──> device radix cache ──> 可选 host / storage tier
                                  └─ prefetch / write-through / 回写确认
```

| 项目 | `MiniRadixCache` | 标准 SGLang 的层级缓存路径 |
|---|---|---|
| 保存的内容 | token prefix 到 slot 的归属关系 | 除 device slot 外，还可跟踪 host/storage 命中与搬运状态 |
| 命中后的动作 | 直接复用 prefix slot | 可能先 prefetch，等待所需层级数据就绪再 admission |
| 淘汰后的影响 | 释放本地 slot | 可能需要 write-through / host tier 的确认，释放时机更复杂 |
| 本文例子 | 默认只解释这一列 | 不展开传输协议；只保留“slot/page 对齐仍成立” |

因此，不要从本节的 `page_size = 4` 示例推导 HiCache 的 host/storage 配置或搬运粒度；它只是讲清本地 KV page 和 slot 关系。真实层级缓存入口见 [`HiRadixCache`](../../python/sglang/srt/mem_cache/hiradix_cache.py#L73) 与 scheduler 的 prefetch 检查（[`get_new_batch_prefill()`](../../python/sglang/srt/managers/scheduler.py#L2814)）。

## 4. Prefill admission 预算

```mermaid
flowchart LR
    Free["allocator free<br/>可立刻分配"] --> Physical["physical tokens<br/>free + evictable"]
    Evictable["cache evictable<br/>可先淘汰"] --> Physical
    Reserve["running decode reserve<br/>不能拿去 prefill"] -->|subtract| Budget["prefill memory budget"]
    Physical --> Budget
    Candidate["candidate cost<br/>extend page round<br/>+ output reserve + 1 page guard"] --> Budget
    Budget --> A[ADMIT]
    Budget --> C[CHUNK]
    Budget --> D[DEFER]
    Budget --> X[ABORT]
```

Prefill admission 要回答的不是“这个 prompt 放不放得下”这么简单，而是：**放它进来后，运行中的 decode 是否还留得出下一页 KV？** 所以它同时受“物理可用内存”“本批 prefill 吞吐上限”和“空闲 request row”三类约束。

### 4.1 预算从哪里来

| 预算 / 限制 | 公式或来源 | 含义 |
|---|---|---|
| `free_tokens` | allocator 的空闲 page × `page_size` | 可立刻给新请求的 KV 容量 |
| `evictable_tokens` | 未锁 radix cache 的 slot | 可以先淘汰，再转给新请求的容量 |
| `physical_tokens` | `free_tokens + evictable_tokens` | 机器实际能凑出的容量上限 |
| `decode_reserved_tokens` | 对每个 RUNNING 请求的未来输出预留，按 page 向上对齐后求和 | 不能被新 prefill 花掉，否则下一轮 decode 容易 OOM |
| `remaining_tokens` | `max(physical_tokens - decode_reserved_tokens, 0)` | admission 真正可花的 KV 内存预算 |
| `remaining_prefill_tokens` | 从 `max_prefill_tokens` 开始，每接收一个请求就扣其新 prefill 长度 | 限制本轮 forward 的计算量，不等同于 KV 内存 |
| `max_new_reqs` | 当前空闲 request row 数 | row 满时，即使 KV 够也只能 `DEFER` |

```text
物理内存： free 16 + evictable 8 = 24 token slots
decode 预留：running-A 需要 4 + running-B 需要 4 = 8
------------------------------------------------------
新 prefill 真正可花：remaining_tokens = 24 - 8 = 16
```

`decode_reserved_tokens` 默认只预留 `remaining_new_tokens × new_token_ratio`；被 retract 过的请求设有 `retracted_stain`，会按 100% 剩余输出预留，避免它重新准入后再次把 decode 推向内存压力。

### 4.2 一个候选请求要花多少

先从 radix cache 求 `prefix_len`。命中的前缀不必重算 KV，因此真正需要 extend 的长度是：

```text
full_extend = total_len - prefix_len
```

候选成本按 page 计算，而不是按 token 精确计算：

```text
candidate_cost = round_page(target - prefix_len)
               + round_page(ceil(remaining_new_tokens × reserve_ratio))  # 仅最后一段
               + page_size                                                # 每个新请求的对齐余量
```

| 成本项 | 为什么存在 |
|---|---|
| `round_page(extend)` | KV allocator 整页分配；写 1 个 token 也可能要拿一页 |
| `output_reserve` | prefill 完后仍要 decode，必须先给未来输出留下空间 |
| `+ page_size` guard | 给每个新请求再留一页对齐余量，避免刚 admission 就因页边界失败 |

例：`page_size = 4`，请求总长 12，命中 cache 前缀 6，剩余输出 4，`new_token_ratio = 0.5`：

```text
extend:         12 - 6 = 6  → round_page(6) = 8
output reserve: ceil(4 × .5) = 2 → round_page(2) = 4
one-page guard:                                      4
------------------------------------------------------
candidate_cost:                                     16
```

只有 `candidate_cost <= remaining_tokens` 才是正常 admission；这个例子正好消耗完上节的 16 token 预算。

### 4.3 四种决策按什么顺序出现

| 结果 | 条件 | 调度器下一步 |
|---|---|---|
| `ADMIT` | 完整 suffix 的成本与本批 prefill 上限都能满足；或完整 cache hit | 本轮完整 prefill，随后可进入 decode |
| `CHUNK` | 完整 suffix 太长，但可放下一个 chunk（优先 `chunked_prefill_size`） | 只跑到 `target_fill_len`，将请求保留为 `chunked_req`，下轮继续 |
| `DEFER` | 当前预算 / row 不够，但以后释放 KV 或完成请求后仍可能运行 | 留在 waiting queue；FCFS 下第一个 defer 会挡住后续 waiting 请求 |
| `ABORT` | 空系统首请求连最小一页都放不下；无 chunk 时整个 extend 也无法放进物理池 | 终止请求，避免永远等待 |

决策顺序如下：

```text
1. 查 radix prefix，计算 full_extend
2. 若固定 chunk 能放下 → CHUNK
3. 若完整 suffix 能放下且未超过本批 prefill 上限 → ADMIT
4. 空系统首请求 / 已开始的 chunk：只要物理 page 放得下，允许 ADMIT 或 CHUNK
   （防止 decode reserve 让系统永久不启动）
5. 尝试从大到小缩短 chunk
6. 连最小 page 都放不下 → ABORT；否则 → DEFER
```

第 4 步是一个刻意的“防饿死”例外：它可以暂时突破保守的 decode reserve，但之后的真实压力会由 eviction → retract → abort 闭环处理；它不是忽略内存上限。

### 4.4 FCFS 与真正分配的边界

`PrefillAdder.add_requests()` 只做**预算决策**，按 FCFS 逐个扣减本轮预算；第一个 waiting 请求因预算 `DEFER` 后，后面请求不能插队。真正的 row 绑定、cache prefix 固定和 slot 分配在 `_get_new_prefill_batch()` / `prepare_for_extend()` 才发生，所以 admission 通过不等于此刻已经写好了 KV 张量。

代码入口：[`MemoryBudget` / `PrefillAdder.__init__()`](../src/my_sglang/schedule_policy.py#L28)、[`_decide()`](../src/my_sglang/schedule_policy.py#L111)、[`_candidate_cost()`](../src/my_sglang/schedule_policy.py#L202)、[`add_requests()`](../src/my_sglang/schedule_policy.py#L80)、[`_get_new_prefill_batch()`](../src/my_sglang/scheduler.py#L199)。测试入口：[`FCFS defer`](../tests/test_schedule_policy.py#L18)、[`cache prefix + evictable`](../tests/test_schedule_policy.py#L40)、[`首个 chunk 防饿死`](../tests/test_schedule_policy.py#L62)、[`retract 后保守预留`](../tests/test_schedule_policy.py#L79)。

### 4.5 真实 SGLang：预算思想相同，策略和公式不逐项等价

第 4.1～4.4 节保留的是“**物理 KV − decode 预留 − 本轮 prefill 上限**”这条主思想，但上面的 `candidate_cost` 与四种结果是教学版的可读模型，不能当作标准 SGLang 的逐行公式。

| 维度 | 教学版 | 标准 SGLang | 学习结论 |
|---|---|---|---|
| 排序 | 固定 FCFS | `SchedulePolicy` 可在 admission 前重排 waiting queue | “第一个 DEFER 阻挡后续”只在教学版 / FCFS 情形成立 |
| decode 预留 | 直接使用固定 `new_token_ratio`；retract 后单请求 `retracted_stain = 100%` | 使用会因 retract 更新、正常 decode 逐步衰减的 `new_token_ratio_tracker` | 都是在避免 prefill 吃掉 decode 空间；调参方式不同 |
| 额外约束 | KV、prefill token、row | 还可受 batch size、priority preemption、LoRA、动态 chunk、HiCache prefetch、disaggregation 等限制 | admission 不是单一内存公式 |
| cache 命中 | 本地 radix 直接得到 `prefix_len` | 可能含 device / storage hit，预取未完成时会暂时跳过请求 | 命中不必然代表本轮立刻能跑 |

```text
教学版：按 FCFS 算一个“可否放入”的清晰预算
真实版：先排序 / 预取 / 检查运行时约束，再用同类 KV 预算挑选可运行集合
```

真实代码证据：[`PrefillAdder` 的运行时参数](../../python/sglang/srt/managers/scheduler.py#L2778)、[`SchedulePolicy.calc_priority()`](../../python/sglang/srt/managers/schedule_policy.py#L126)、[`HiCache prefetch 检查`](../../python/sglang/srt/managers/scheduler.py#L2814)、[`new_token_ratio_tracker` 的 retract/decay 更新](../../python/sglang/srt/managers/scheduler.py#L3010)。

## 5. Decode 压力与 unfinished chunk cache

```mermaid
flowchart TD
    Need[计算 decode 新 page] --> Free{free 足够?}
    Free -->|no| Evict[evict 未锁 radix 叶子]
    Evict --> Again{足够?}
    Again -->|no, 多请求| Retract[retract victim → waiting]
    Again -->|no, 最后请求| Abort[abort]
    Free -->|yes| Run[prepare_for_decode]
    Again -->|yes| Run
```

压力闭环在 [`_get_decode_batch()`](../src/my_sglang/scheduler.py#L276)。另一方面，chunk 每次 forward 成功后，[`_cache_unfinished_req()`](../src/my_sglang/scheduler.py#L394) 会把 committed 的完整 page 插入 radix tree 并保持 lock，未满页尾部仍归请求。

测试入口：[`retract/re-admit`](../tests/test_scheduler.py#L304)、[`last-request abort`](../tests/test_scheduler.py#L334)、[`unfinished chunk cache`](../tests/test_scheduler.py#L228)。

### 5.1 真实 SGLang：同样是闭环，资源与反馈更多

教学版故意把压力处理缩成“本地 KV page 不够”。真实 SGLang 的闭环仍是 `检查 → 释放可驱逐 cache → retract → 最后 abort`，但每一步管理的资源更多：层级 cache 的 write-through 状态、Mamba 等额外 allocator、指标/流式 abort 消息，以及下一轮 admission 的全局预留比例。

| 维度 | 教学版 | 标准 SGLang |
|---|---|---|
| 释放对象 | `MiniRadixCache` 的可驱逐本地 page | KV、可驱逐 cache，以及按模型配置存在的额外资源 |
| retract 后的预算反馈 | 被 retract 的请求带 `retracted_stain`，下次按 100% 预留 | `new_token_ratio_tracker` 根据这轮 retract 更新，后续成功 decode 再逐步衰减 |
| abort 可见性 | 标记请求结束 | 同时向 tokenizer / 客户端发送 abort 输出，并记录 metrics |
| unfinished chunk cache | 成功后仅缓存完整本地 page | 还可能等待层级 cache write-through 确认，才成为可驱逐空间 |

因此，`retract victim → waiting` 在本文中只表示“释放教学版的运行态 KV 后重新 admission”；它不等价于真实服务中所有资源都同步、立即释放完毕。真实实现的 KV 压力处理见 [`update_running_batch()`](../../python/sglang/srt/managers/scheduler.py#L3010)，而层级缓存会在 prefill 路径处理 write-through ack（[`get_new_batch_prefill()`](../../python/sglang/srt/managers/scheduler.py#L2998)）。

## 6. Overlap 两阶段

核心不是“同时调度两批请求”，而是把一批的 **分配/提交** 拆开：

```text
launch_step()                         finalize_pending()
─────────────                         ──────────────────
schedule + allocate + start/kick      等 runner 取回 token
             │                                  │
             ▼                                  ▼
allocated 已推进，committed 未推进       committed 追上 allocated
             │                                  │
             └──── PendingOverlapStep ──────────┘
```

所以 pending 窗口内的正确不变量是 `kv_allocated_len > kv_committed_len`；只有 runner 成功返回 token 后才能 commit。这个类**禁止** pending 期间再次调用 `launch_step()`，因此 overlap 的边界是“已 kick 的 runner 工作”和调用方后续协调之间，而不是调度器自行发出下一批。

```mermaid
sequenceDiagram
    participant Caller
    participant S as MiniOverlapScheduler
    participant R as LazyRunner
    Caller->>S: launch_step()
    S->>S: settle + schedule + prepare allocation
    alt first prefill
        S->>R: prefill_start + prefill_kick
    else chunked extend
        S->>R: extend_start + extend_kick
    else decode batch
        S->>R: decode_batch_start + decode_batch_kick
    end
    S->>S: save PendingOverlapStep
    Note over S,R: pending: allocated > committed
    S-->>Caller: OverlapLaunchResult
    Caller->>S: finalize_pending()
    S->>R: finalize（取回 / 等待 token）
    R-->>S: token
    S->>S: commit + process result + cache/state transition
    Note over S,R: allocated == committed
    S-->>Caller: StepResult
```

### 6.1 两阶段各自做什么

| 阶段 | 做了什么 | 请求的关键状态 |
|---|---|---|
| `launch_step()` | settle 上轮；选 EXTEND 或 DECODE；`prepare_for_*()` 分配 slot 并写入映射表；runner `start + kick`；保存 pending | `allocated` 已增加；`committed` 仍是旧值；还没有 `output_ids` |
| `finalize_pending()` 成功 | runner `finalize` 取回 token；`commit_allocated()`；处理 token、finish、radix cache 与 batch 状态 | `committed == allocated`；请求继续 RUNNING 或 FINISHED |
| 任一阶段异常 | 清除 pending；回滚未提交 slot；移除 runner 请求；请求 reset 后回 waiting | 不留半分配状态；下次可重新 admission |

`PendingOverlapStep` 是两阶段之间的“收据”，保存本批 `batch`、给调用方看的 `forward` 快照、EXTEND 的每请求 handle 或 DECODE 的批 handle，以及本轮已发生的 retract / abort rid。它确保 finalize 的对象就是 launch 时已分配的对象。

| batch 模式 | launch 调用 | finalize 调用 |
|---|---|---|
| 首次 prefill | `prefill_start()` → `prefill_kick()` | `prefill_finalize()` |
| 后续 chunked prefill | `extend_start()` → `extend_kick()` | `extend_finalize()` |
| decode | `decode_batch_start()` → `decode_batch_kick()` | `decode_batch_finalize()` |

### 6.2 用一个请求看 `allocated` 与 `committed`

下面对应测试中的 2-token prompt；`slot` 已在 launch 阶段分配，但模型 token 还没有拿到，所以不能提前 commit：

```text
第 1 轮：EXTEND / prompt 长度 2
launch:    allocated = 2, committed = 0, runner = prefill_start + kick
finalize:  token = 10 → allocated = 2, committed = 2, output_ids = [10]

第 2 轮：DECODE / 新增一个位置
launch:    allocated = 3, committed = 2, runner = decode_start + kick
finalize:  token = 11 → allocated = 3, committed = 3, output_ids = [10, 11]
```

这正是 “allocation 可以先发生，模型结果成功后才提交” 的含义。若 `finalize` 返回数量不对或抛异常，代码不会执行 `commit_allocated()`，而是走 `rollback_uncommitted()` 和重新入队；不能把 `allocated` 当成模型已经完成的证据。

### 6.3 调用规则与普通 `step()` 的关系

| 调用方式 | 行为 |
|---|---|
| `launch_step()` → `finalize_pending()` | 显式暴露 pending 窗口；必须成对调用 |
| 只调用 `finalize_pending()` | 报错：没有已 launch 的批 |
| pending 时再调用 `launch_step()` | 报错：必须先 finalize 当前批 |
| `step()` | 只是顺序调用 `launch_step()` 再 `finalize_pending()`；不会留下给调用方的 pending 窗口 |

### 6.4 标准 SGLang 与教学版：相同主线，不同并发模型

> **不要把 `MiniOverlapScheduler` 当成真实 SGLang overlap 的逐行翻版。** 两者都在解决“不要让 CPU 结果处理白白挡住 GPU forward”，但教学版把它收缩成一个可见的 `launch → finalize` 事务；真实 SGLang 在内部 event loop 中用 `result_queue`、CUDA stream / event 和 batch copy 做流水。

```text
教学版（调用方必须串行）
CPU:  launch B0 ── pending B0 ── finalize B0 ── launch B1
GPU:       B0 forward                 B1 forward

标准 SGLang（内部流水；结果处理滞后一批）
CPU:  选 B0 / 提交 B0 ── 选 B1 / 提交 B1 ── 处理 B0 result ── 采样 B1
GPU:          B0 forward ───────────────────── B1 forward
                     ↑ CPU 侧调度/结果处理与 GPU 工作重叠
```

上图表达的是 **CPU 调度、结果处理与 GPU 计算重叠**；不是“任意多批 GPU forward 同时执行”。真实实现还会因为连续 prefill、`spec + grammar` 等条件主动关闭某一轮 overlap，保证正确性或 TTFT。

| 对比点 | `MiniOverlapScheduler`（教学版） | 标准 SGLang（当前源码） | 学习时应怎么理解 |
|---|---|---|---|
| 驱动者 | 调用方显式 `launch_step()` / `finalize_pending()` | Scheduler 内部 `event_loop_overlap()` | 教学版把内部时序拆成两个可单测 API |
| 上一批结果放哪里 | 单个 `PendingOverlapStep` | `result_queue` 中的 `(batch.copy(), batch_result)` | 真实版复制 batch，避免后续调度修改影响上一批结果处理 |
| 何时选下一批 | 必须 finalize 当前 pending 后 | **先** `get_next_batch_to_run()` 和 `run_batch(current)`，再处理上一批 result | 真实版能让 CPU 的结果处理与当前 GPU 工作重叠 |
| GPU / CPU 协调 | `start + kick + finalize` 的 runner handle | `run_batch()`、`FutureMap`、CUDA forward/copy stream、event/barrier | 教学版只保留“已提交但未取结果”的概念，不模拟设备同步细节 |
| 普通 KV 长度推进 | launch 时只推进 `allocated`；finalize 成功才推进 `committed` | 普通 EXTEND / DECODE 的 `prepare_for_*()` 会同时推进 `kv_committed_len` 和 `kv_allocated_len` | **两者字段时序不同，不能拿教学版的 `allocated > committed` 当成标准规则** |
| `allocated > committed` 的主要来源 | 每一个 pending batch 都会出现 | 主要用于 speculative decoding 等 over-allocation；普通 1-token decode 通常两者一起 `+1` | 它在真实版表示“多分配但未被接受的 KV”，不等同于“GPU 尚未返回” |
| 特殊禁用条件 | 没有建模 | 连续 prefill、`spec + grammar` 等会关闭 overlap | 生产调度首先保证依赖正确，再争取重叠 |

#### 为什么真实版能“先提交 B1，再处理 B0”？

真实 loop 的顺序是：取 B1 → `run_batch(B1)` → 将 `B1.copy()` 与 result 入队 → `process_batch_result(B0)` → 为 B1 采样。这使 B0 的 CPU 结果处理尽量与 B1 的设备执行重叠。它不是把 B0 的 `ScheduleBatch` 留在一个公开 pending API 中等调用方 finalize。

```mermaid
sequenceDiagram
    participant S as 标准 SGLang Scheduler
    participant Q as result_queue
    participant G as GPU / worker
    S->>S: get_next_batch_to_run(B1)
    S->>G: run_batch(B1)
    S->>Q: append(B1.copy(), result)
    Note over S,G: B1 的设备工作进行中
    S->>Q: pop(B0, result)
    S->>S: process_batch_result(B0)
    S->>S: launch_batch_sample_if_needed(B1)
```

#### `allocated` / `committed` 的差异，必须分开记

```text
教学版（人为建立事务边界）
prepare / launch: allocated = 7, committed = 4
finalize 成功:     allocated = 7, committed = 7

标准 SGLang，普通 EXTEND / DECODE
prepare:           allocated = 7, committed = 7
结果处理:           追加 output、finish、stream、cache / 释放等

标准 SGLang，speculative decode（可能一次预留多个候选位置）
prepare:           allocated > committed
verify / 接受后:   committed 按实际接受 token 推进；多余部分可释放
```

因此，本节的 `commit_allocated()` / `rollback_uncommitted()` 应理解为教学版为了把“物理预占”和“模型成功”画清楚而加入的**显式事务边界**。真实 SGLang 的普通路径依赖 batch copy、队列、设备 stream/event 和更细的 release 逻辑，不存在同名的 `finalize_pending()` 作为统一提交点。

真实代码证据：[`event_loop_overlap()`](../../python/sglang/srt/managers/scheduler.py#L1535)、[`is_disable_overlap_for_batch()`](../../python/sglang/srt/managers/scheduler.py#L1594)、[`init_overlap()`](../../python/sglang/srt/managers/scheduler.py#L1220)、[`ScheduleBatch.prepare_for_extend()`](../../python/sglang/srt/managers/schedule_batch.py#L2004)、[`ScheduleBatch.prepare_for_decode()`](../../python/sglang/srt/managers/schedule_batch.py#L2602)、[`mix_with_running()`](../../python/sglang/srt/managers/schedule_batch.py#L2377)。

代码入口：[`launch_step()`](../src/my_sglang/overlap_scheduler.py#L83)、[`finalize_pending()`](../src/my_sglang/overlap_scheduler.py#L151)、[`rollback_uncommitted()`](../src/my_sglang/schedule_batch.py#L136)。测试入口：[`allocated/committed 与调用顺序`](../tests/test_overlap_scheduler.py#L90)、[`chunked + radix + paged allocator 组合`](../tests/test_overlap_scheduler.py#L132)。

## 与真实 SGLang 的对应关系

| my-sglang | SGLang 对应概念 | 可以迁移的主线 | 不能直接迁移的部分 |
|---|---|---|---|
| `MiniScheduler` | `Scheduler.get_next_batch_to_run()` | prefill 优先、last/running batch | 真实可策略排序、MIXED batch，且有运行时外围条件 |
| `MiniScheduleBatch` | `ScheduleBatch` | extend/decode allocation 与过滤 | 真实携带 GPU tensor、采样、多模态、推测解码等状态 |
| `PrefillAdder` | `schedule_policy.PrefillAdder` | free、evictable、decode reserve 的预算思想 | 固定 FCFS / 简化公式不等价于真实策略组合 |
| `ReqToTokenPool` | `mem_cache.memory_pool.ReqToTokenPool` | 二维逻辑位置到 slot 映射 | CPU NumPy vs device torch tensor |
| paged allocator | `mem_cache.allocator` | page 对齐、尾页复用、释放 | 后端与模型类型会影响真实分配细节 |
| `MiniRadixCache` | `RadixCache` / `HiRadixCache` | prefix ownership、lock、LRU | 无 host/storage tier、prefetch、write-through |
| `MiniOverlapScheduler` | overlap event loop | “结果延迟 / 异步边界”的学习模型 | 并非真实 result queue、CUDA stream/event 的逐行等价 |

继续阅读：[数据结构、所有权与不变量](data-structures.md) → [带数字的动态流程](dynamic-flows.md)。
