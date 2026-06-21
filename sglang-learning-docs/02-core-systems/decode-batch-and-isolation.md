# Decode Batch、请求隔离与 MIXED

> 本篇关注 `running_batch -> ScheduleBatch(DECODE)`、逐 token KV 分配、attention/sampling 隔离，以及 EXTEND 与 DECODE 的混合执行。
>
> 上一篇：[Prefill Batch 与 KV 分配](./prefill-batch-and-kv.md) · 总索引：[Req 到 ScheduleBatch](./request-batch-state-flow.md)

## `running_batch` 到 `ScheduleBatch(DECODE)`

如果本轮没有可运行的新 prefill batch，Scheduler 会继续推进已有请求：

```text
new_batch = get_new_batch_prefill()

if new_batch is None and running_batch is not empty:
  running_batch = update_running_batch(running_batch)
  return running_batch
```

`update_running_batch()` 的主线是：

```text
1. filter_batch(): 删除 finished/abort 请求
2. check_decode_mem(): 检查下一轮 KV 空间，必要时 retract
3. prepare_for_decode(): 准备本轮输入和 KV 位置
```

普通 decode 的 `ScheduleBatch(DECODE)` 通常不是新对象，而是被准备好的 `running_batch`。

`prepare_for_decode()` 会为每个请求推进一个位置：

```text
forward_mode = DECODE
每个 req 分配 1 个新 KV slot
ReqToTokenPool[req_pool_idx, old_seq_len] = new_slot
更新 seq_lens 和 KV committed/allocated 状态
```

例如：

```text
running_batch.reqs = [ReqA, ReqB, ReqC]
req_pool_indices   = [2, 5, 9]
seq_lens           = [6, 4, 5]
out_cache_loc      = [80, 81, 82]

ReqA row 2, next token_pos -> slot 80
ReqB row 5, next token_pos -> slot 81
ReqC row 9, next token_pos -> slot 82
```

Decode 通常只把每个请求最新生成的 token 作为 query 输入。历史 token 不需要重新进入完整 `input_ids`，attention 会通过 `ReqToTokenPool -> TokenToKVPool` 读取历史 K/V。

## Decode batch 里多个 Req 为什么不会互相影响

`ScheduleBatch(DECODE)` 把多个请求放进同一次 GPU forward，但 batch 维只是并行容器。请求边界由以下行级字段共同表达：

```text
batch.reqs[i]             = 第 i 行对应哪个 Req
batch.req_pool_indices[i] = 该请求在 ReqToTokenPool 的行
batch.seq_lens[i]         = 该请求当前可见历史长度
batch.out_cache_loc[i]    = 本轮新 token 的 KV 写入位置
```

示例：

```text
row i:              0      1      2
req:              ReqA   ReqB   ReqC
req_pool_indices:   2      5      9
seq_lens:           8      4      6
out_cache_loc:      80     81     82
```

`ReqToTokenPool` 中每个请求有独立行：

```text
row 2 / ReqA: slots [10,11,12,13,14,15,16,17]
row 5 / ReqB: slots [30,31,32,33]
row 9 / ReqC: slots [50,51,52,53,54,55]
```

Attention backend 为每个 query row 构造自己的 KV index list：

```text
qA -> row 2, token_pos [0,8) -> ReqA slots only
qB -> row 5, token_pos [0,4) -> ReqB slots only
qC -> row 9, token_pos [0,6) -> ReqC slots only
```

逻辑上等价于 block-diagonal mask：

```text
                 KV slots
             ReqA       ReqB       ReqC
           +----------+----------+----------+
qA         | visible  | blocked  | blocked  |
qB         | blocked  | visible  | blocked  |
qC         | blocked  | blocked  | visible  |
           +----------+----------+----------+
```

实现通常不会真的生成这个巨大 mask，而是通过 paged KV、ragged metadata、page table、`kv_indices` 或 `cu_seqlens_k` 限制每个 query 能读取的 slots。

## EXTEND 与 DECODE 的 attention 差异

EXTEND 中一个请求可能有多个新 token，因此既需要请求间隔离，也需要请求内部 causal 关系：

```text
ReqA suffix = [a3, a4, a5]

a3 sees prefix + a3
a4 sees prefix + a3 + a4
a5 sees prefix + a3 + a4 + a5
```

逻辑结构：

```text
          keys / values
          ReqA       ReqB       ReqC
query  +---------+---------+---------+
ReqA   | causal  | masked  | masked  |
ReqB   | masked  | causal  | masked  |
ReqC   | masked  | masked  | causal  |
```

DECODE 中每个普通请求通常只有一个最新 query token，因此 backend 更常直接使用 per-request KV index list，而不是构造完整 block causal mask。

## `ForwardBatch` 如何传递请求边界

`ForwardBatch.init_new()` 从 `ScheduleBatch` 获得关键字段：

```text
req_pool_indices = batch.req_pool_indices
seq_lens         = batch.seq_lens
out_cache_loc    = batch.out_cache_loc
```

不同 backend 将它们变成自己的执行 metadata：

| Backend | 典型 metadata | 作用 |
|---|---|---|
| Triton | `kv_indptr` / `kv_indices` | 指定每行 query 的 KV index 范围 |
| FlashAttention | `cu_seqlens_k` / `page_table` | 表达 ragged 长度和 paged KV |
| FlashInfer | paged decode wrapper plan | 为不同 `seq_lens` 规划 decode |

这些 metadata 才是“第 i 个 query 只能读取第 i 个请求历史”的实际执行边界。

## Sampling 为什么也不会串

Attention 输出后的 logits 仍按 batch row 对齐：

```text
logits[0] -> ReqA
logits[1] -> ReqB
logits[2] -> ReqC
```

`SamplingBatchInfo` 按相同的 `ScheduleBatch.reqs` 顺序构造，因此每行使用自己的 sampling params、penalty、grammar 和 logprob 配置。只要行级映射保持一致，attention 和 sampling 都不会跨请求串状态。

## MIXED：把 Prefill 和 Decode 放进同一轮

启用 mixed chunked prefill 且条件允许时，Scheduler 可以把新请求 EXTEND 和老请求 DECODE 合并：

```text
new_batch     = ScheduleBatch(EXTEND) from waiting_queue
running_batch = existing decode requests

running_batch.prepare_for_decode()
new_batch.mix_with_running(running_batch)
return new_batch as MIXED
```

合并后的结构大致是：

```text
forward_mode  = MIXED
reqs          = new prefill reqs + old decode reqs
extend_lens   = prefill suffix lengths + [1,1,...]
out_cache_loc = prefill new slots + decode new slots
```

可以把它理解为：

```text
MIXED = EXTEND flat token 区域 + DECODE 每请求一个 token 的区域
```

它的目标是在限制长 prefill 干扰的同时，让已有 decode 请求继续推进。请求间隔离仍由每行的 request/KV metadata 保证。

## Decode 内存不足与 retract

每轮 decode 都要为存活请求增加 KV。如果 `check_decode_mem()` 发现空间不足，Scheduler 可以 retract 部分请求：

```text
running_batch
  -> 选择需要 retract 的 Req
  -> 释放/回收相应 KV 占用
  -> Req 重新进入 waiting_queue
  -> 调高未来输出占用估计
```

这不是常规公平调度，而是 admission 预测过于激进或负载变化时的安全兜底。

## 常见误解

| 误解 | 正确理解 |
|---|---|
| `ScheduleBatch(DECODE)` 一定是新对象 | 普通路径通常就是准备后的 `running_batch` |
| Decode 不需要历史上下文 | 需要；只是从 KV cache 读取，不再重传完整 token 序列 |
| Batch 中的请求会互相 attend | 不会；每个 query row 有自己的 KV index/page table 范围 |
| 必须创建一个巨大 block mask | 通常通过 ragged/paged metadata 隐式表达边界 |
| Sampling 只靠请求顺序“碰巧”对应 | `SamplingBatchInfo` 与 batch 行级映射一起维护对应关系 |
| `running_batch` 始终正在 GPU 执行 | 它是长期逻辑状态；新 prefill 运行时它可以暂时不执行 |

## 源码阅读地图

| 主题 | 文件 / 函数 |
|---|---|
| 更新 running | `scheduler.py:update_running_batch` |
| 准备 DECODE | `schedule_batch.py:ScheduleBatch.prepare_for_decode` |
| DECODE KV 分配 | `mem_cache/common.py:alloc_for_decode` |
| Scheduler→ModelRunner | `model_executor/forward_batch_info.py:ForwardBatch.init_new` |
| Attention metadata | `layers/attention/*_backend.py:init_forward_metadata` |
| Sampling 行级状态 | `sampling/sampling_batch_info.py` |
| MIXED 合并 | `schedule_batch.py:ScheduleBatch.mix_with_running` |
| Retract | `schedule_batch.py:ScheduleBatch.retract_decode` |

回到：[Req 到 ScheduleBatch：状态流转导读](./request-batch-state-flow.md)
