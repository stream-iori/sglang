# 多卡：按输出拆、按输入拆，最后怎么合并

> 源码基准：learning `2aa70e3eb4`，已合入 origin/main `6fc8d9da32`（2026-10-10）。

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

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/layers/linear.py](../../python/sglang/srt/layers/linear.py) | ColumnParallelLinear / RowParallelLinear |
| [python/sglang/srt/models/qwen3.py](../../python/sglang/srt/models/qwen3.py) | QKV、o_proj、MLP 组合 |
| [python/sglang/srt/distributed/parallel_state.py](../../python/sglang/srt/distributed/parallel_state.py) | 通信组 |
| [python/sglang/srt/runtime_context.py](../../python/sglang/srt/runtime_context.py) | Attention/MoE rank 派生 |
