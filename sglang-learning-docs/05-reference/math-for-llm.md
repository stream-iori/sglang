# 读推理代码需要的矩阵和概率

> 源码基准：上游 `16eb87c06f`；本地集成 `4f0e1cbea2`（2026-10-20）。本地日志增强与验证范围见总索引和 Mac 验证记录。

| 对象 | 常见 shape | 含义 |
|---|---|---|
| input_ids | [N] | 本轮 N 个 token 的词表索引 |
| X | [N, hidden] | embedding/残差表示 |
| W | [input_dim, output_dim] | 本页的数学记法；框架存储可能转置 |
| Q/K/V | [N, heads, head_dim] 等 | Attention 向量 |
| logits | [请求数, vocab] 或其他选中位置布局 | 每个候选 ID 的分数 |

## 矩阵乘

```text
Y = XW
[N, input_dim] × [input_dim, output_dim] → [N, output_dim]
```

共享维度相同才能乘。按输出拆产生不同输出片段；按输入拆产生需要相加的贡献，见 [多卡](../03-advanced/multi-gpu.md)。

## Attention

```text
scores = Q @ Kᵀ / sqrt(head_dim)
weights = softmax(scores + mask)
output = weights @ V
```

mask 限制未来位置/窗口等。GQA 中 Q heads 与 KV heads 的对应由 backend 处理，不要求数目相同。

## 分数与采样

假设 logits=[2,1,0]：先减最大值防止指数过大，再 exp 并归一化。temperature 会改变分布锐利程度；top-k/top-p 会限制候选，具体处理顺序读 sampler。

greedy 选择最大分数。概率采样需要随机数，所以输入相同不保证输出相同；比较执行正确性时要明确 seed、sampling 配置、数值容差。

## 性能估算

矩阵乘运算量可粗估为 `2×N×input_dim×output_dim` FLOPs。这个估算不包含 Attention、激活、读写、通信和 kernel launch，不能直接等同于端到端耗时。

## shape、索引与 view 先于公式

| 操作 | 例子 | 含义 |
|---|---|---|
| batch 维 | `[5,1024]` | 本轮 5 个有效 token，各有 1024 特征 |
| slice | `X[0:2]` | 选 A 的两个位置，不是复制完整上下文 |
| gather | 按请求映射找 KV | 索引值决定读取位置，不要求连续 |
| reshape/view | `[N,H,D]` 与 `[N,H*D]` | 表达变化；是否共享底层存储还看具体操作 |
| transpose | 交换轴 | 不能因此假定数据已经连续搬运 |

同样的 shape 可以有不同 stride、dtype、device 和地址。Graph 和 Host 传输尤其关心这些差异，不能只看 print(shape)。

## 一个手算 Attention

取一个 query，对两个历史 key 的缩放分数为 `[0,0]`，无 mask，两个 value 分别 `[2,0]`、`[0,4]`。softmax 权重为 `[0.5,0.5]`，输出 `[1,2]`。若第二个位置被 mask 成负无穷，权重变成 `[1,0]`，输出 `[2,0]`。

这个例子说明：同一组 Q/K/V，边界或 mask 错了，输出仍可能是合法数值，却已经串上下文。

## logits 不等于概率

```python
import math
logits = [2.0, 1.0, 0.0]
exps = [math.exp(x - max(logits)) for x in logits]
probs = [x / sum(exps) for x in exps]
assert abs(sum(probs) - 1) < 1e-12
assert probs[0] > probs[1] > probs[2]
```

自测：把三个 logits 都加 100，softmax 不变；把它们乘 2，一般会更尖锐。temperature 改的是相对差距，不是给文字附加“创造力标签”。

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/layers/linear.py](../../python/sglang/srt/layers/linear.py) | 矩阵投影 |
| [python/sglang/srt/layers/sampler.py](../../python/sglang/srt/layers/sampler.py) | 采样 |
| [python/sglang/srt/layers/attention/torch_native_backend.py](../../python/sglang/srt/layers/attention/torch_native_backend.py) | Attention 参考执行 |
