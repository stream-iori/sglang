# 标准 SRT（一）：真实执行链

[返回全局地图](README.md) · 上一页：[教学 overlap](05-overlap.md) · 下一页：[持续组批与显存](07-standard-scheduling.md)

**tiny runner 中的一次模型调用，在标准 SRT 中拆成“组织执行、准备设备输入、运行模型、调用算子后端”几层。** 层次变多，仍在回答总图 03～04 的问题：输入什么、历史在哪里、新 KV 写哪里。

本轮三篇以仓库 `a5a7123f54` 的基础自回归文本生成路径为证据。默认从单 Worker、普通 Attention、无 speculative、无 DiT 的路径理解；图是职责关系，不要求每个框都对应独立进程。

## 图 S1：把总图 03～04 放大

```text
01 Scheduler：挑请求、准入、维护生命周期
           |
02～03 ScheduleBatch：本轮请求、输入、KV 地址、采样参数
           |
       TpModelWorker：组织一次 generation 调用
           |
       ForwardBatch.init_new：准备执行所需的设备数据与元数据
           |
04     ModelRunner：选择并组织模型执行路径
           |
       模型 forward：Embedding -> 各层 Attention / FFN -> logits
           |                          |
           |                    Attention backend
           |                    准备元数据、调用 kernel、访问 KV
           v
       sampling -> GenerationBatchResult -> 05 处理结果
```

| 层 | 负责什么 | 用一个问题检验边界 |
|---|---|---|
| Scheduler / ScheduleBatch | 组批、分配和请求状态 | “A 这轮能不能参与？” |
| TpModelWorker | 把调度工作接到模型执行与采样 | “这批生成请求怎样得到结果对象？” |
| ForwardBatch | 传递本轮模型调用需要的数据 | “每请求多长，新增 token 写哪些 slots？” |
| ModelRunner | 持有模型与执行资源，组织 forward | “这批走什么执行路径？” |
| 模型模块 | 定义层与张量计算 | “hidden 怎样变成 logits？” |
| Attention backend | 将 Attention 的数学需求落到具体设备实现 | “怎样从分页 KV 读出历史？” |

当前 `TpModelWorker.forward_batch_generation()` 直接接收 `ScheduleBatch` 并构造 `ForwardBatch`。不要为了套旧版本的层次图，额外补一个本路径不存在的 `ModelWorkerBatch` 中间步骤。

证据：[`TpModelWorker.forward_batch_generation()`](../../../python/sglang/srt/managers/tp_worker.py)、[`ModelRunner.forward()`](../../../python/sglang/srt/model_executor/model_runner.py)。

## 图 S2：Attention 数学与物理地址之间还有一层

```text
逻辑上下文：A 的 pos 0,1,2
                 |
地址表：req_to_token[row_A,:3] = [8,9,12]
                 |
ForwardBatch：row、seq_len、新 KV 地址、positions 等
                 |
backend 元数据：把请求边界、历史地址组织成 kernel 需要的布局
                 |
Attention kernel：读取 KV[8]、KV[9]、KV[12]，与当前 Q 计算
```

历史在逻辑上连续，并不要求在物理 slot 上连续。不同 backend 可使用不同的索引、页表或工作区形式；不能把某一个 backend 的元数据结构当成全 SRT 的统一格式。

| 已有知识 | 标准实现新增的一层 |
|---|---|
| `QKᵀ → softmax → V` | kernel 如何并行和分块执行 |
| row/pos 找 slot | backend 如何将批内多请求映射整理给 kernel |
| prefill 与 decode 输入长度不同 | backend 按模式准备元数据和执行路径 |

证据：[`AttentionBackend.init_forward_metadata()`](../../../python/sglang/srt/layers/attention/base_attn_backend.py)、[`ForwardBatch`](../../../python/sglang/srt/model_executor/forward_batch_info.py)。

## 启动做一次，本轮计算反复做

```text
启动准备：模型配置 / 权重 / 设备与通信 / 内存池 / 执行后端 / 可选 graph capture
                                  |
                                  v
反复执行：收请求 -> 组批 -> 更新本轮输入与元数据 -> forward -> 处理结果
```

上图列出依赖资源，不是所有配置下的严格初始化调用顺序。权重通常常驻设备，不为每次 decode 重新加载；KV 则随着请求推进被写入、共享和回收。

CUDA Graph 可把一段设备工作捕获后重放，减少重复提交的开销。它复用的是执行过程，**不是上次生成结果**；新输入仍要填入对应缓冲区，满足形状与路径条件才能 replay。

| 方式 | 每轮仍要做 | 可以复用什么 |
|---|---|---|
| 普通执行 | 准备输入、提交计算、获取结果 | 权重、KV 池等常驻资源 |
| Graph replay | 更新输入与必要元数据，执行本轮计算 | 已捕获的工作及固定缓冲区；可按支持的 batch size padding |

本篇先认识 Graph 的位置，capture 细节属于后续专题。证据：[`load_model()`](../../../python/sglang/srt/model_executor/model_runner.py)、[`DecodeCudaGraphRunner`](../../../python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py)。

## 一个必须分开的 prefix 命中特例

| 路径 | prefix 命中后的首 token 从哪里来 |
|---|---|
| my-sglang tiny | 允许完整命中；额外存最后 hidden state，用它得到 logits |
| 当前标准 Req 默认上限 | `_compute_max_prefix_len()` 限制为 `input_len - 1`，保留末位置用于计算；logprob 要求还能缩短匹配范围 |

页对齐等条件可能使实际命中更短。不要将 tiny 的额外 hidden 缓存当作标准 KV cache 的天然能力。证据：[`Req._compute_max_prefix_len()`](../../../python/sglang/srt/managers/schedule_batch.py)。

## 合上文档自检

| 问题 | 答案要点 |
|---|---|
| 为什么有了物理 slot 映射，还需要 Attention backend？ | 映射说明数据在哪里；backend 将它组织成设备 kernel 能执行的元数据与计算。 |
| Graph replay 会复用旧 token 输出吗？ | 不会；复用执行过程，新输入仍产生新的计算结果。 |
| tiny 与标准模型执行是否只是 NumPy 换成 PyTorch？ | 还多了执行资源、后端适配、设备数据与批处理结果等职责。 |
