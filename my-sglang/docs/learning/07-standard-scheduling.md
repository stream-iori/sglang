# 标准 SRT（二）：持续组批与显存预算

[返回全局地图](README.md) · 上一页：[真实执行链](06-standard-execution.md) · 下一页：[采样到响应](08-standard-output.md)

**调度器同时控制两件事：这一轮算多少，长期内存装不装得下。** 对应总图 01、02、06。单条请求的 token 因果不变，多请求让“什么时候轮到谁”成为新问题。

## 图 S3：持续组批，每轮成员可以变化

```text
时间向右 ------------------------------------------------->
A：  prefill    decode    decode    结束
B：             prefill   decode    decode    decode ...
C：                                 prefill   decode ...

每轮：收新请求 -> 选择可执行请求 -> forward -> 移除完成者 -> 下一轮
```

上图是各请求生命周期示意，不代表每一列一定由一个混合 batch 执行。持续组批的核心是允许请求陆续加入和退出，不必等整批最慢请求结束才收下一批。

| 工作 | 改变什么 | 不改变什么 |
|---|---|---|
| 接收新请求 | waiting 候选增加 | 已运行请求的上下文 |
| 准入新请求 | 新请求拿到 row/KV，参与 prefill | 旧请求已确认的输出 |
| 过滤完成请求 | batch 成员与行顺序 | 其他请求的身份及历史 |
| 合并 batch | 本轮输入与参数的排列 | 一个请求内的自回归依赖 |

模型输入、row、长度、采样参数都必须跟随请求一起过滤和合并。batch 中的下标不是稳定 request row。

证据：[`Scheduler.event_loop_normal()` / `get_next_batch_to_run()`](../../../python/sglang/srt/managers/scheduler.py)、[`ScheduleBatch`](../../../python/sglang/srt/managers/schedule_batch.py)。

## 图 S4：新 prompt 与已有 decode 怎样共享时间

```text
纯分开执行： [新请求 prefill chunk] -> [老请求 decode] -> ...
              老请求等待本轮结束

支持 mixed 时的一批： [新请求的多个 prompt token | 老请求各一个 decode 输入]
                         extend 部分               decode 部分
```

| 概念 | 解决的问题 | 不要混淆 |
|---|---|---|
| chunked prefill | 把长 prompt 拆成较小的计算段 | 不等于自动混入 decode |
| mixed chunk | 在满足条件时把 prefill 与已有 decode 输入组织到同一批 | 不是所有配置都启用 |
| overlap | 错开 CPU 处理与设备执行 | 不等于 mixed batch |

当前标准实现检查 `is_mixed_chunk`、running batch 非空等条件，并排除部分不兼容情况，例如此分支的 return_logprob；符合时调用 `mix_with_running()`。因此“有 prefill 就不做 decode”只适合教学简化路径。

取舍是：长 prefill 可以延长老请求等待；切块或混合有机会改善等待，但也会改变执行效率。不能只靠开关名称断言吞吐必然提升。

证据：[`Scheduler` 的 Mixed-style chunked prefill 分支](../../../python/sglang/srt/managers/scheduler.py)。

## 显存容量从哪里来

```text
设备可用空间
  ├─ 常驻权重及其他已分配资源
  ├─ 为运行时留出的余量：激活、工作区等
  └─ 可用于 KV 的预算 -> 按每 token KV 字节数换算 -> 页对齐与上限约束
```

标准配置器先观察当前可用显存，已经常驻的权重等自然占掉了其中一部分，再扣运行时预留。当前代码中的 slack 使用 `pre_model_load_memory × (1 - mem_fraction_static)`；它不是简单把“GPU 标称显存 × 比例”全部交给 KV。

对普通、未量化、K/V head dimension 相同的 Attention，先用以下公式建立量级感：

```text
每 token KV 字节数 = 2 × L × Hkv × Dhead × dtype_bytes
                    K/V  层数 KV头数  每头维度   每元素字节
```

这是所有这些层的 KV 合计。多卡时应使用当前设备实际持有的层与 KV heads，不能不考虑复制情况就机械除以卡数。

| 手算参数（示例，不是当前模型配置） | 值 |
|---|---:|
| L | 32 |
| Hkv | 8 |
| Dhead | 128 |
| dtype_bytes | 2 |
| 每 token KV | 131072 bytes = 128 KiB |
| 假设 KV 预算 1 GiB | 理想容量 8192 tokens |

实际还要考虑页对齐、padding、池布局及用户上限；8192 是这个简化例子的容量，不是某张 GPU 的承诺。降低 KV heads 或元素字节数会改变 KV 成本；**权重量化也不自动等于 KV 用相同精度**。

证据：[`KVCacheConfigurator._profile_available_bytes()` / `config_from_budget()`](../../../python/sglang/srt/mem_cache/kv_cache_configurator.py)、[内存池容量计算](../../../python/sglang/srt/model_executor/pool_configurator.py)。

## 图 S5：总容量门与单轮计算门

```text
新请求候选
    |
    +-> 容量门：row、可用/可回收 KV、已有请求后续生成预留
    |
    +-> 本轮门：未命中 suffix 有多少，chunk / prefill token 预算够不够
    |
    v
完整加入 / 只加一段 / 等待 / 无法执行
```

假设本轮 prefill 预算 4 tokens，请求上下文长度 7、已有可复用 prefix 4：本轮完整 suffix 只有 3，能通过本轮计算门。容量门仍需检查新增页、已有缓存占用和后续生成预留。

即使 decode 每请求只输入一个 token，序列越长，需要读取的历史 KV 也越多。输入数少不代表成本恒定。

## 用三个观察项理解调度取舍

| 观察项 | 流程含义 | 可能相关的原因 |
|---|---|---|
| 首 token 延迟 TTFT | 从选定起点到首个输出 | 排队、tokenize、prefill、输出传输 |
| 后续输出间隔 | 相邻输出之间的等待 | decode 执行、调度让路、输出缓冲 |
| 吞吐 | 单位时间完成多少 token/请求 | 批大小、输入输出长度、cache 命中、后端效率 |

客户端观察值与服务内部计时起止点可能不同；SSE 消息也未必一条对应一个 token，不能把消息间隔直接当纯 GPU decode 时间。指标定义见 [`metrics_collector.py`](../../../python/sglang/srt/observability/metrics_collector.py)。

## 合上文档自检

| 问题 | 答案要点 |
|---|---|
| KV 内存够，为何仍把 prompt 分块？ | 本轮计算量和等待时间也需要控制。 |
| 一批新增 3 个 token，是否只占 3 个 token 的 KV？ | 历史 KV 仍占用，新页还有尾部空位；还需考虑后续生成。 |
| mixed 与 overlap 能否用一个概念替代？ | 不能，一个改变批内组成，一个改变执行与结果处理的交接时序。 |
