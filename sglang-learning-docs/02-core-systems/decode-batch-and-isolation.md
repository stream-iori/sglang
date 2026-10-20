# Decode 与请求隔离：拼在一起也不串上下文

> 源码基准：上游 `16eb87c06f`；本地集成 `4f0e1cbea2`（2026-10-20）。本地日志增强与验证范围见总索引和 Mac 验证记录。

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

## 首个输出和首个 decode 的区别

```text
prefill 输入 [p0 p1 p2] → 写 p0/p1/p2 的 KV → logits → 采样 y0
下一轮输入 [y0]        → 写 y0 的 KV       → logits → 采样 y1
下一轮输入 [y1]        → 写 y1 的 KV       → logits → 采样 y2
```

若采样出 y2 后请求结束，普通路径可能不再计算 y2 的 KV。所以 `输入 token 数 + 输出 token 数` 不能不看阶段就当作已写入 KV 的数量。投机验证、预留容量、分页又会改变计数方式。

## 两条隔离线都要成立

| 隔离线 | 实际职责 | 如何取证 |
|---|---|---|
| query 边界 | A、B 的新增输入分开切片 | extend_seq_lens、positions |
| KV 访问 | 每个 query 只读允许的历史 | req_pool_indices、seq_lens、mask、KVLocPlan |

只核对 packed token 顺序不足以证明隔离；只核对请求行不同也不足以证明正确 mask。可复用前缀允许共享只读历史，后缀写入和可变 SSM 状态仍需独立。

## 本地验收

`verify_mac.py` 同时发送两个不同输入，并与各自串行 greedy 输出 ID 比较，另发请求确认服务继续工作。通过只证明当前 MPS 配置下这些请求输出一致；实际是否合进同一个设备 batch，要结合日志判断。

不能通过查看生成文字“看上去不同”来证明请求隔离，也不能宣称两次 HTTP 并发必然触发同批。

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/layers/attention/torch_native_backend.py](../../python/sglang/srt/layers/attention/torch_native_backend.py) | 逐请求 start_q/end_q 与 KV 读取 |
| [python/sglang/srt/managers/schedule_batch.py](../../python/sglang/srt/managers/schedule_batch.py) | prepare_for_decode / filter_batch |
| [python/sglang/srt/mem_cache/kv_loc_plan.py](../../python/sglang/srt/mem_cache/kv_loc_plan.py) | 读表和写地址 |
