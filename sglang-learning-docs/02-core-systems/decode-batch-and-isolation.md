# Decode 与请求隔离：拼在一起也不串上下文

> 源码基准：learning `2aa70e3eb4`，已合入 origin/main `6fc8d9da32`（2026-10-10）。

普通 decode 每个活跃请求通常贡献一个新输入位置，用已有 KV 预测下一个 token。

## 同一批内的边界

```text
请求 A：本轮新增 3 个 token    请求 B：本轮新增 2 个 token
packed input = [a0 a1 a2 b0 b1]
offset       = [0, 3, 5]

请求索引 0：offset[0]:offset[1] = 0:3 → A 的 query
请求索引 1：offset[1]:offset[2] = 3:5 → B 的 query
```

`offset` 是解释用的累加边界，不是说所有 Attention backend 都有同名字段。当前 torch_native 用 `extend_seq_lens` 累加 `start_q/end_q`，用请求自己的 KV 映射和序列长度选择历史上下文。

| 元数据 | 防止什么错误 |
|---|---|
| extend_seq_lens / 起始位置 | 把 A 的新增 query 读成 B 的 |
| req_pool_indices | 找错请求映射行 |
| seq_lens / prefix 长度 | 读取未有效写入的位置 |
| Attention mask | 读到未来位置或不允许的范围 |
| KVLocPlan / 翻译表 | 把逻辑槽位误用于别的子池 |

物理槽位不要求相邻。共享的是合法匹配的前缀；分叉之后的写位置必须独立，SSM 等可变状态还要考虑 copy-on-write。

## 一次 decode 的准备

```text
过滤已结束请求 → 确认容量 → 分配本轮写槽
    → 上轮生成 ID 作为本轮输入 → 更新长度/positions
    → 模型计算/采样 → 追加新 ID → 结束检查
```

token 已生成，不表示它的 KV 一定已经写入：普通自回归路径中，采样出来的 ID 通常要在下一轮成为输入后才获得自己的 KV。

## 混合批与 padding

MIXED 可以把 prefill 的新增片段和 decode 位置拼在一轮。打包有效 token 可以减少无用计算，但 Graph、某些 backend、分布式 collective 对 shape 有额外要求，仍可能需要 padding。

padding 不负责表达真实请求边界；长度、索引和 mask 才负责。见 [Graph 详解](cuda-graph-and-padding.md)。

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/layers/attention/torch_native_backend.py](../../python/sglang/srt/layers/attention/torch_native_backend.py) | 逐请求 start_q/end_q 与 KV 读取 |
| [python/sglang/srt/managers/schedule_batch.py](../../python/sglang/srt/managers/schedule_batch.py) | prepare_for_decode / filter_batch |
| [python/sglang/srt/mem_cache/kv_loc_plan.py](../../python/sglang/srt/mem_cache/kv_loc_plan.py) | 读表和写地址 |
