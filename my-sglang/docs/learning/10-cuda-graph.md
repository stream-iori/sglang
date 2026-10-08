# 标准 SRT（五）：CUDA Graph 怎样接入逐轮执行

[返回全局地图](README.md) · 上一页：[启动与 IPC](09-startup-ipc.md) · 下一页：[TP 与性能观察](11-tp-performance.md)

**Graph 复用提交工作的方式，KV cache 复用历史计算结果，两者省掉的工作不同。** 本篇放大总图 04，以普通 decode 的 CUDA Graph 路径为例；源码基准为 `a5a7123f54`。

## 图 S10：普通提交与 replay

```text
普通执行每轮：CPU 提交算子 A -> 提交 B -> 提交 C -> GPU 计算

准备阶段：    捕获兼容的一段工作 A -> B -> C
                                  |
之后每轮：更新本轮输入缓冲区 -> replay -> GPU 仍执行 A、B、C
```

| 对象 | 复用什么 | 每轮仍在变化什么 |
|---|---|---|
| 模型权重 | 参数 | 输入与激活 |
| KV cache | 已处理 token 的 K/V | 新 token 的 K/V 与读取范围 |
| CUDA Graph | 被捕获的设备工作与缓冲区安排 | 输入 token、请求映射、长度及结果 |
| overlap | CPU/设备工作交接的流水安排 | 每批数据与完成事件 |

Graph 不会让有前后依赖的 token 同时产生，也不保证任何 batch 都能走 replay。

## 图 S11：变化的 batch 怎样使用捕获档位

假设演示配置捕获了 batch size 1、2、4、8，支持 padding；这些数字只用于手算。

```text
实际 batch_size=3
       |
检查本轮模式、特性、后端是否兼容
       |
选择支持的档位 4
       |
缓冲区：[请求 A | 请求 B | 请求 C | padding]
       |
更新输入与所需元数据 -> replay -> 只向真实请求交付有效结果
```

| 条件 | 结果 |
|---|---|
| 实际大小和特性可由已捕获路径支持 | 可以 replay |
| 允许 padding 且存在合适档位 | 可使用带 padding 的执行布局 |
| 禁止 padding | 需满足后端对相应 graph key 的支持条件 |
| 大小超出范围或存在不兼容特性 | 不能使用这条 Graph 路径，通常转普通执行 |

padding 是占位执行，不是新用户请求。不能把 padding 输出追加到任何真实 Req，也不能破坏它的 KV。

证据：[`DecodeCudaGraphRunner.can_run_graph()` 与输入准备](../../../python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py)。真实检查还包括模式与 backend 条件，不仅是比较 batch_size。

## 为什么要把元数据准备分成 graph 内外

```text
ForwardBatch
     |
graph 外准备：依赖 host / 动态形状等不能捕获的工作
     |
graph 内准备：适合捕获的静态形状设备操作
     |
Attention / 模型计算
```

当前 AttentionBackend 明确分开 `init_forward_metadata_out_graph()` 与 `init_forward_metadata_in_graph()`。后者不应执行 `.item()`、`.cpu()`、`.tolist()` 等 host 同步操作或动态形状分配。

原因是：捕获与重放依赖可复用的执行结构；把每轮 Python 决策混入捕获区域，并不能让 Python 随 replay 自动重新执行。

证据：[AttentionBackend 的元数据契约](../../../python/sglang/srt/layers/attention/base_attn_backend.py)。

## 性能与资源的取舍

| 收益来源 | 对应成本或限制 |
|---|---|
| 减少重复提交开销 | 捕获需要启动时间与资源 |
| 复用固定缓冲区 | 需要更新输入，受支持形状与特性限制 |
| padding 覆盖更多实际大小 | 可能带来额外占位计算 |

如果瓶颈主要是历史 KV 读取或通信，减少 CPU 提交开销不等于同比减少整轮耗时。是否获益应在相同请求长度、并发和后端下实测，不能用 Fake CUDA 的时序图估算。

## 合上文档自检

| 问题 | 答案要点 |
|---|---|
| 前一轮 input 是 10，replay 下一轮会不会仍算 10？ | 输入缓冲区必须更新为本轮值；Graph 不是结果缓存。 |
| 实际 3 个请求，按 4 执行，是不是生成 4 个用户答案？ | 不是，额外一行是 padding。 |
| 所有 Python 工作都能放入 Graph 吗？ | 不能，host 决策、同步与动态形状工作需遵守捕获边界。 |
