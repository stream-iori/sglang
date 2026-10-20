# 多卡：按输出拆、按输入拆，最后怎么合并

> 源码基准：上游 `16eb87c06f`；本地集成 `4f0e1cbea2`（2026-10-20）。本地日志增强与验证范围见总索引和 Mac 验证记录。

以 `Y = XW` 为例：X 是 `[tokens, input_dim]`，W 是 `[input_dim, output_dim]`。按输入/输出拆指的是这个数学维度，不是 HTTP 输入/输出请求。

## 按输出维度拆：ColumnParallelLinear

```text
W = [W0 | W1]                     output_dim 分成两份
GPU0：Y0 = X @ W0
GPU1：Y1 = X @ W1
完整 Y = [Y0 | Y1]                可以拼接，也可以继续保持分片
```

各 GPU 使用完整输入特征，计算部分输出特征。常用于 Q/K/V 投影、MLP 的扩展投影（如 gate/up），具体取决于模型实现。

## 按输入维度拆：RowParallelLinear

```text
X = [X0 | X1]，W = [W0; W1]       input_dim 分成两份
GPU0：P0 = X0 @ W0
GPU1：P1 = X1 @ W1
完整 Y = P0 + P1                  求和归约，不是拼接
```

各 GPU 计算同一输出的部分贡献。常用于 Attention 输出投影、MLP down 投影。前一层若保持了输出分片，后一层可以直接接收对应的输入分片。

| 并行方式 | 拆什么 | 常见通信 |
|---|---|---|
| TP | 单层张量/矩阵 | all-reduce、all-gather、reduce-scatter 等 |
| DP | 请求由不同副本处理 | 路由、副本各自执行 |
| attention DP/CP | Attention 请求或上下文维度 | 具体 group 和同步策略 |
| EP | MoE 专家 | token dispatch/combine |
| PP | 模型层 | stage 间激活传递 |

当前 rank/width 的派生在 runtime_context 和 parallel_state。尤其 Attention/MoE 的 rank 不一定就是简单的 tp_rank，应该查派生公式和实际 group。

## 为什么多卡可能更慢

```text
总时间 ≈ 分片计算 + 通信 + 同步等待 + 调度/提交
```

小模型、小 batch 下通信可能比省下的计算更贵。KV heads 少于 TP 规模时还要检查复制/分片策略，不能直接把每 token KV 字节机械除以 TP。

本地 MPS 是单设备验证；这里是源码与数学讲解，不是已经完成的多卡 benchmark。

## 一个可以手算的分片例子

令 X 为 `[2,4]`，W 为 `[4,6]`，两卡：

| 分法 | 每卡计算 | 还原结果 |
|---|---|---|
| 按输出拆 | `[2,4] @ [4,3] → [2,3]` | 拼接两个输出，得 `[2,6]` |
| 按输入拆 | `[2,2] @ [2,6] → [2,6]` | 两份部分和相加，得 `[2,6]` |

“Column/Row”相对权重张量的逻辑约定，框架内部实际存储可能转置；读 layer 的 forward 和通信代码确认，别只看内存 shape。

## rank 不是一个整数解释所有并行

同一进程可以参与不同通信组。TP、Attention TP/DP/CP、MoE TP/EP、PP 的 rank/size 各有含义，当前配置还涉及 CP/TP group sharing、KV 分片等规则。

```text
全局进程集合
    ├─ 用于某次 Attention 的通信组
    ├─ 用于专家 dispatch/combine 的组
    └─ 流水线 stage 的上下游
```

核对 group 成员、张量切分轴、collective 顺序，再推导结果；跨 rank 顺序不一致可能挂住，而不是简单报维度错误。

## 硬件实验验收

先固定单卡基线，再用同模型 TP2 比较输出、每卡容量和耗时。记录 workload 和通信时间；若变慢，先拆开计算与同步开销。小模型 TP2 不一定比单卡快，不能把卡数增加当作成功标准。

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/layers/linear.py](../../python/sglang/srt/layers/linear.py) | ColumnParallelLinear / RowParallelLinear |
| [python/sglang/srt/models/qwen3.py](../../python/sglang/srt/models/qwen3.py) | QKV、o_proj、MLP 组合 |
| [python/sglang/srt/distributed/parallel_state.py](../../python/sglang/srt/distributed/parallel_state.py) | 通信组 |
| [python/sglang/srt/runtime_context.py](../../python/sglang/srt/runtime_context.py) | Attention/MoE rank 派生 |
