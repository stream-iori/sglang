# 关键对象与字段图谱

**先确定对象负责什么，再沿流程观察字段怎样变化。** 六张图以 my-sglang 为主，标准 SRT 差异在每图下说明。这里的“模型”既包括请求、batch 等数据对象，也包括执行预测的 Transformer，二者分别标注。

[返回学习入口](README.md) · [可执行状态账本](12-request-ledger.md) · [完整字段参考](../data-structures.md)

```text
F1 对象总关系
 ├─ F2 Req：状态与确认历史
 ├─ F3 地址与 KV 水位 -> F4 工作单与模型执行
 ├─ F5 异步交接
 └─ F6 缓存所有权与回收
```

蓝色：请求与确认输出；绿色：KV；紫色：模型；橙色：异步交接；红色：回滚或回收。箭头表示流程或读写，不表示类继承。图片内省略的精确条件以图下说明为准。

六张图均为 2048×1152 PNG，可点击图片查看大图。原图经等比例放大与补白交付，详见[提示词与尺寸记录](assets/field-atlas/prompts.md)。

## F1：对象分层——长期记录、本轮工作与执行资源

![F1 核心对象总关系图](assets/field-atlas/f1-objects.png)

读图：从 Scheduler 选 Req 开始，沿 batch 到 runner，再沿结果回到 Req；最后看底部地址与资源层。

| 对象 | 保存/负责什么 | 谁更新、谁使用 |
|---|---|---|
| Req | 一条请求跨轮的输入、确认输出、状态、资源关联 | Scheduler 更新，组批和结果处理读取 |
| waiting_queue / chunked_req / running_batch | 等待、未完成分块、可 decode 的请求归属 | Scheduler 选择和交接 |
| MiniScheduleBatch | 可变的本轮准备单，包括新分配和输入 | prepare 写入，commit/rollback 处理 |
| ForwardBatch | runner 使用的本轮输入与地址元数据 | 从准备单构造，runner 读取 |
| allocator / ReqToTokenPool | 分配资源 / 逻辑位置到物理 slot 的地址表 | Scheduler 管理，模型通过映射读历史 |
| KV pool | token 的 K/V 向量；tiny 使用 NumPy | 模型写入新 K/V，Attention 读取 |
| radix cache | 可共享 prefix 的 slot 与锁引用 | 请求借用、交接和淘汰 |
| FutureMap / result_queue | 执行侧下一轮输入 / 尚待 CPU 处理的工作结果 | 异步执行与 CPU 按不同时间点使用 |

流程：选请求 → 准备本轮范围与资源 → 构造 ForwardBatch → 执行 → 确认结果 → 更新状态或回收。`last_batch` 是等待容器交接的上一批，不能与异步 `result_queue` 混为一谈。

ForwardBatch 固定了数值元数据，但 `reqs` 仍引用可变 Req，并非深拷贝所有请求状态。图中 tiny 模型计算与 Fake CUDA overlap 是两种教学路径，当前不能直接组合使用。

**标准对照：** 标准 Scheduler、TpModelWorker、ModelRunner、Attention backend 将执行职责拆得更细；参考[真实执行链](06-standard-execution.md)。

## F2：Req——状态与输出历史

![F2 Req 状态与生成历史](assets/field-atlas/f2-request.png)

读图：横向看每轮动作，纵向比较同一时刻的 prompt、输出列表与组合视图。例子为同步路径，输出上限 3，预设输出 10、11、12。

| 字段/视图 | 含义 | 写入或计算时机 | 使用者 |
|---|---|---|---|
| `rid` | 请求身份 | 创建请求时 | 队列、runner 和结果关联 |
| `origin_input_ids` | 原始 prompt | 创建时 | prefill 与重建 |
| `output_ids` | CPU 已确认的生成 token 列表 | 结果处理接受新 token 时追加 | 下一轮同步输入、停止检查、retract 重建 |
| `get_fill_ids()` | prompt + 已确认输出 | 调用时派生，不是另一份长期账本 | EXTEND 输入范围 |
| `status` | 教学请求状态 | chunk、结果、retract、finish 处理时 | 调度资格判断 |
| `sampling_params.max_new_tokens` | 新输出上限 | 请求创建时 | 停止判断、预算 |
| `eos_token_ids` / `stop_token_ids` | 模型 EOS / 用户停止编号 | 创建或配置时 | 结果停止判断 |
| `finished_reason` | 长度、停止 token 或 abort 原因 | 终止时 | 结果报告 |

EXTEND 输入 `[7,8]` 后确认 10；DECODE 输入 10 后确认 11；输入 11 后确认 12，达到上限结束。新输出 12 不需要再写 KV。`output_ids` 是列表，`len(output_ids)` 才是数量，`remaining_new_tokens` 是剩余数量的派生值。

长 prompt 的中间 chunk 只补 KV，不能确认完整 prompt 后的答案。短 prompt 一次 EXTEND 完成，不会停在 PREFILLING；首输出立即满足停止条件时可以直接 FINISHED。

**标准对照：** 标准没有同名统一 `status` 枚举，需结合容器、finish reason 与 retract 标记。overlap 时 CPU 状态可滞后，例如已 launch 的教学请求仍标 WAITING，见 F5 与[实跑账本](12-request-ledger.md)。

代码：[`Req`](../../src/my_sglang/models.py)、[`结果处理`](../../src/my_sglang/scheduler.py)。

## F3：地址与水位——token 编号不是 KV 地址

![F3 逻辑 token 到物理 KV](assets/field-atlas/f3-memory.png)

读图：左边把 pos、token、slot 对齐，右边沿同步准备→成功/失败读水位变化。主快照是 decode 输入 10 成功后、请求未结束。

| 字段/对象 | 精确含义 | 谁写 / 谁读 |
|---|---|---|
| `req_pool_idx` | 请求当前活跃地址表行号 | admission 分配，映射及 FutureMap 使用 |
| `req_to_token[row,pos]` | 逻辑位置对应的物理 slot | prepare/缓存绑定写入，模型读取 |
| `kv.kv_allocated_len` | 已分配位置的长度边界 | prepare 推进，回滚或释放重置 |
| `kv_committed_len` | 调度器已提交的 KV 边界 | commit 推进，回收重置 |
| page / slot | 分配单位 / token 的 K/V 地址位置 | allocator 管理，模型使用 |

例子：row0 的 pos2 是 token10，映射到 slot4。Attention 读取的是该位置的 K/V 向量，而不是把整数 10 当向量。3 个映射位置占用 2 页、4 slots，尾部 slot5 尚未使用。

| 路径 | committed 前进时机 | 不能据此推断什么 |
|---|---|---|
| 同步 | runner 成功返回后 | 新采样 token 已经有 KV |
| overlap | 异步 launch 成功后 | 设备执行完成、复制完成或 CPU 输出确认 |

失败回滚要清除 `[committed,allocated)` 的映射，并释放不与稳定历史共页的新增页，再将 allocated 拉回 committed。若新位置复用旧尾页，不可将旧页一并释放。

**标准对照：** 当前标准 `Req.kv` 下包含三个字段：`kv_allocated_len`、`kv_committed_len`、`cache_protected_len`；教学后两个仍在 Req 顶层。标准 request row0 为 padding，图中 row0 是教学可用行。

代码：[`prepare / commit / rollback`](../../src/my_sglang/schedule_batch.py)、[`内存池`](../../src/my_sglang/pools.py)。

## F4：工作单——字段怎样成为模型行为

![F4 ForwardBatch 与模型执行](assets/field-atlas/f4-forward.png)

读图：先看 A、B 各自的新增范围，再看展平字段，最后看写 KV 和读历史两条支路。此图是独立双请求 EXTEND 示例：A 只需扩展 suffix，B 从新 prompt 开始。

| ForwardBatch 字段 | 示例 | 模型使用方式 |
|---|---|---|
| `forward_mode` | EXTEND | 区分本轮扩展或 decode |
| `input_ids` | `[10,4,5]` | Embedding 的新输入 |
| `req_pool_indices` | `[0,1]` | 找 A、B 的地址表行 |
| `out_cache_loc` | `[4,6,7]` | 与输入逐项对齐，定位新 K/V 写入 |
| `seq_lens` | `[3,2]` | 各请求本轮结束后的总长度 |
| `extend_seq_lens` | `[1,2]` | 将展平输入切为 A 的 1 个与 B 的 2 个 |
| `extend_range_starts` | `[2,0]` | 各请求本轮起点，用于正确位置与上下文 |
| `prefix_indices_by_req` | 各请求借用的缓存 slots | 教学观察字段；不要认为它列出所有历史 KV |
| `contains_last_prefill_chunk` | 是否全部到最后段 | 区分完整 prompt 与未完成 chunk 的批级信息 |

`MiniScheduleBatch.extend_lens` 在 ForwardBatch 中名为 `extend_seq_lens`。`prefix_len + extend_len = seq_len`；A 为 `2+1=3`，B 为 `0+2=2`。

模型按 token 编号做 Embedding，按 position 处理 RoPE，按 out_cache_loc 写新 K/V，按地址表读取可见历史。随后 Attention、FFN、最终归一化与 LM head 产生 logits，sampling 选 token。图省略部分 Norm、残差和投影，只强调字段如何驱动执行。

数值相同不等于含义相同：B 输入的 token4 与 A 写入的 slot4，处于不同编号空间。模型输出先作为结果返回，再由 Scheduler 判断是否可以追加到 output_ids。

**标准对照：** 标准 backend 还需准备自己的 Attention 元数据，执行层见[真实执行链](06-standard-execution.md)；不能把教学字段直接当作所有 backend 的页表格式。

代码：[`ForwardBatch`](../../src/my_sglang/models.py)、[`tiny 模型`](../../src/my_sglang/tiny_transformer.py)。

## F5：异步交接——三个 token 观察点

![F5 overlap 三处 token 记录](assets/field-atlas/f5-overlap.png)

图底部三处记录展示不同观察时刻：队列中的 B0 尚待处理，`output_ids` 展示 process 后追加的结果；不是同一时刻的联合快照。

读图：先沿 CPU 一行看提交与确认，再沿 forward 看 token 依赖，最后看三个记录的不同用途。B0/B1 指稳定阶段 decode，不是首次 prefill。

| 对象/字段 | 内容 | 生产者 → 消费者 |
|---|---|---|
| `FutureMap.output_tokens_buf[row]` | 下次 decode 要读的 token | 执行侧 stash → 后继 gather |
| `valid[row]` | 教学调试的一次消费标志 | stash=True，gather 后=False |
| `result_queue` | job、batch 快照、异步 result | launch 入队 → CPU FIFO 处理 |
| `copy_done` | host 结果可读的完成门槛 | copy/event → CPU 等待 |
| `Req.output_ids` | 已确认输出历史 | CPU 处理有效结果时追加 |

```text
B0 sample10 -> stash10 -> B1 gather10 -> B1 sample11
                  |
             B0 D2H -> copy_done -> CPU 确认10
```

两条支路有各自依赖。图不是精确耗时刻度；Fake CUDA 等 B0.copy_done 只推进必要前缀，不会顺便执行 B1。真实设备可自主推进已提交工作。这里的 `row` 是稳定活跃请求行，不是本轮 batch 下标。

job 已入队时，token 可能尚未产生；token 已 stash 时，CPU 可能尚未确认；host 副本是否可读仍看 copy_done。不能把三个观察点合成一个“已经生成”。

**标准对照：** 标准 FutureMap 的 token stash 与 `publish()` 中其他元数据交接不能混用；valid 消费检查不是所有生产配置的必备字段。教学 pipeline 异常重排也不是标准任意 CUDA 故障的保证。

代码：[`FutureMap / pipeline`](../../src/my_sglang/overlap_scheduler.py)、[`FakeCudaRunner`](../../src/my_sglang/runner.py)。用[实跑账本](12-request-ledger.md)核对 CPU 快照与实际 gather 的差异。

## F6：所有权——共享页、私有页与重建

![F6 缓存所有权与回收](assets/field-atlas/f6-ownership.png)

读图：上部先找共享页与私有尾页，下部按 evict、retract、finish 三条路径看“谁放弃了什么”。此图为独立共享前缀示例，启用 radix、page_size=2。

| 字段/对象 | 保存什么 | 如何变化 |
|---|---|---|
| `prefix_indices` | 当前借用的 cache slots | admission 匹配，分块交接后可更新 |
| `last_node` | 被锁住路径的终点 | 用来解除/迁移路径的锁引用 |
| `lock_ref` | cache 路径的活跃保护引用 | 借用加锁，释放减锁 |
| `cache_protected_len` | 请求已受 cache 保护的前缀长度 | 跟随匹配与交接更新；本身不是锁操作 |
| `retracted_stain` | 曾因压力撤回 | retract 置 True，再准入更保守预留 |
| `_inflight_refs` / `_deferred_finished` | 教学在途引用计数 / 待释放结束请求 | 入队增加、结果结算减少，最后引用结束才释放 |

| 动作 | 清理什么 | 保留什么 |
|---|---|---|
| evict | 无锁 cache leaf 的页 | 活跃请求锁定的页 |
| retract | 活跃 row、私有 KV、runner 状态，解除借用 | prompt、确认输出、已缓存前缀仍可由 cache 持有 |
| finish | 结束请求；启用 cache 时交接可靠完整页，回收其余资源 | 已确认结果，以及 cache 保留的完整页 |

`retract` 后 `get_fill_ids()` 仍是 prompt+output_ids；重新匹配和分配可换 row/slots。重算旧 token 的 KV 不等于再次向用户输出旧 token。finish 时若后继 batch 尚在途，先保持资源，结算并丢弃多算结果，再释放。

**标准对照：** 标准保护长度位于 `req.kv.cache_protected_len`。标准生命周期及 cache 策略更复杂，不将教学 `_inflight_refs` 当成一一对应字段；页保护与避免过早复用的目的仍适用。

代码：[`缓存交接 / 回收`](../../src/my_sglang/scheduler.py)、[`radix`](../../src/my_sglang/radix_cache.py)、[`延迟释放`](../../src/my_sglang/overlap_scheduler.py)。可运行[三请求案例](13-shared-kv.md)逐页检查。

## 跨图的标准字段对照

| 教学路径 | 当前标准 SRT 路径 | 注意 |
|---|---|---|
| `req.output_ids` | `req.output_ids` | 已确认 token 历史；列表与标准容器实现不同 |
| `req.kv.kv_allocated_len` | `req.kv.kv_allocated_len` | 分配边界职责对应 |
| `req.kv_committed_len` | `req.kv.kv_committed_len` | 标准层级不同；具体提交时机需按路径核对 |
| `req.cache_protected_len` | `req.kv.cache_protected_len` | 标准层级不同 |
| `req.status` | 容器、finish reason、retract 等共同表达 | 无统一同名状态枚举 |

核对依据：[教学模型定义](../../src/my_sglang/models.py)、[标准 Req/ReqKvInfo](../../../python/sglang/srt/managers/schedule_batch.py)。配图生成方式、提示词与尺寸见[生成记录](assets/field-atlas/prompts.md)。
