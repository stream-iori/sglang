# my-sglang 运行时复习题

先读[全局图文学习路线](../docs/learning/README.md)，再用这里的题目检查对应节点。

## 答案口径修正

以下说明优先于旧表中的简化措辞：

| 易混点 | 准确口径 |
|---|---|
| committed 时机 | 同步在 runner 成功返回后提交；overlap 在异步 launch 成功后提交，不保证设备计算已经完成。 |
| output_ids | 它是已确认 token 列表；`len(output_ids)` 才是数量。 |
| 展平字段拆分 | `MiniScheduleBatch.extend_lens`；传给 runner 的 `ForwardBatch.extend_seq_lens`。 |
| token 与 KV | 同步下一轮开始前，最后一个输出尚无 KV；overlap 需结合后继 batch 是否已执行判断。 |
| cache 保护 | `lock_ref` 实施引用保护；`cache_protected_len` 记录保护长度。 |
| 分块后的状态 | 最后 chunk 通常进入 RUNNING；若首输出立即达到停止条件则 FINISHED。 |

## 先记住这条因果链

```text
模型输入 token T  -> 写入 KV -> 预测新 token T+1

已确认输出： [10, 11]
下一次输入：             11
本轮写 KV：              11
本轮新输出：                  12
下轮才写 KV：                12
```

## 第一组：8 道主线题

| 题 | 问题 | 标准答案 |
|---:|---|---|
| 1 | `RequestStatus.PREFILLING` 和一次 `ForwardMode.EXTEND` 有何区别？为什么短 prompt 做 EXTEND，却通常不会停在 PREFILLING？ | `EXTEND` 是**本轮 forward 模式**：处理 prompt 或未完成的 prompt chunk。`PREFILLING` 是**请求跨轮状态**：说明 prompt 还未全部写入 KV。短 prompt 虽做一次 EXTEND，但该轮已完成全部 prompt，所以直接进入 `RUNNING`。`DECODE` 才是逐 token 推进的 forward 模式。 |
| 2 | prompt=`[7,8]`，prefill 产生 `10`，下一轮 decode 产生 `11`。第二轮 forward 的输入是什么？第二轮结束后，`11` 是否已写进 KV？ | 第二轮输入是 `10`。该轮会把 `10` 写入 KV，并产生 `11`。`11` 是新 output，尚未写入 KV；若请求未结束，要等下一轮把 `11` 作为输入时才写入。 |
| 3 | `kv_allocated_len`、`kv_committed_len`、`len(output_ids)` 分别代表什么？为什么 overlap 下三者可能不同？ | `kv_allocated_len` 是已申请物理 slot 的边界；`kv_committed_len` 是调度器已提交的边界，同步在 runner 成功后、overlap 在 launch 成功后推进；`len(output_ids)` 是 CPU 已确认的输出数。overlap 可先提交后继工作，稍后才确认旧结果；launch 本身不保证 FutureMap 已写入或设备计算已完成。 |
| 4 | `FutureMap.output_tokens_buf[row]` 和 `ReqToTokenPool.req_to_token[row,pos]` 都使用 request row。它们各自存什么，为什么不能合并？ | `FutureMap[row]` 存**一个 token**：下一次 decode 的模型输入。`ReqToTokenPool[row,pos]` 存**一个物理 KV slot**：逻辑序列位置 `pos` 的 K/V 地址。一个负责 token relay，一个负责 K/V 定位，维度和生命周期不同。 |
| 5 | overlap 的稳定 decode 为什么必须遵守 `launch B1 -> process B0`？B1 是否会在 B0 前真正完成 forward？ | B1 先入 forward FIFO，能在 GPU 侧从 `FutureMap[row]` 取 B0 产生的 token；然后 CPU 才等待并处理 B0，提交 `output_ids`。B1 只是排队，不能越过同一 FIFO stream 上的 B0 完成 forward。 |
| 6 | chunked prefill：prompt=`[1,2,3,4,5]`，chunk size=`2`。前三次 EXTEND 后，请求状态和 `output_ids` 是什么？哪个 chunk 的返回 token 才可对外提交？ | 第 1 轮输入 `[1,2]`：`PREFILLING`、`[]`；第 2 轮 `[3,4]`：`PREFILLING`、`[]`；第 3 轮 `[5]`：`RUNNING`、`[首 token]`。只有最后一个 chunk 的返回 token 可以提交，因为此前 prompt 还不完整。 |
| 7 | radix cache 命中 `[1,2,3,4]`，新请求是 `[1,2,3,4,9]`。本轮 `extend_range` 是什么？命中的 KV 是复制、重新计算，还是借用？它靠什么避免被淘汰？ | `extend_range=[4,5)`，只计算 `9`。命中 KV 是借用已有 slot，既不复制也不重算。活跃请求通过 radix node 的 `lock_ref` 与请求的 `cache_protected_len` 保护该前缀，避免被 LRU 淘汰。 |
| 8 | decode KV 不够时，`evict -> retract -> abort` 各释放什么？retract 后哪些数据必须保留，下一次 admission 为什么用 `prompt + 已确认 output_ids` 重建？ | `evict`：释放无锁的 cache leaf page，活跃请求继续运行。`retract`：释放一个 `RUNNING` 请求的 row、私有 KV、runner 状态；保留逻辑输出。`abort`：最后仍无法推进时结束请求。retract 后使用 `prompt + 已确认 output_ids` 重建，保证已给用户的 token 不被重采样。 |

## 第二组：巩固题

| 题 | 问题 | 标准答案 |
|---:|---|---|
| 1 | prompt=`[7,8]`，输出已有 `[10,11]`，请求未结束。下一次 DECODE 输入是什么？此时 KV 应包含哪些 token？ | 输入是 `11`。这轮开始前，KV 包含 `[7,8,10]`；本轮把 `11` 写入 KV，并产出新 token（例如 `12`）。本轮结束、未释放时 KV 才包含 `[7,8,10,11]`。 |
| 2 | 用一句话分别定义：`kv_allocated_len`、`kv_committed_len`、`output_ids`。 | `allocated` 是已拿到 slot 的边界；`committed` 是调度器已提交的 KV 边界，提交时机见上表；`output_ids` 是 CPU 已确认的生成 token 列表。 |
| 3 | 为什么 `FutureMap[row]` 只存一个 token，`ReqToTokenPool[row,:]` 却存一串 slot？ | 每次 decode 只需要上一个生成 token 作为单个输入，所以 FutureMap 每个 row 只需一个 relay token。完整上下文的每个逻辑 token 都有 K/V，因此请求行要按位置保存一串物理 KV slot。 |
| 4 | 分块 prefill 的中间 chunk 为什么不能把 runner 返回 token 写进 `output_ids`？ | 中间 chunk 只覆盖不完整 prompt；它最后位置的 logits 对应的是尚未补全的上下文，不能当作完整 prompt 后的首个生成 token。只有最后 chunk 完成全部 prompt 后，返回 token 才可见。 |

## 最小状态图

```text
短 prompt： WAITING --完整 EXTEND--> RUNNING --DECODE...--> FINISHED
长 prompt： WAITING --非最后 EXTEND--> PREFILLING --最后 EXTEND--> RUNNING
KV 紧张：   RUNNING --retract--> WAITING --重新 EXTEND--> RUNNING
```
