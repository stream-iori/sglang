# 读推理代码需要的矩阵和概率

> 源码基准：learning `2aa70e3eb4`，已合入 origin/main `6fc8d9da32`（2026-10-10）。

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

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/layers/linear.py](../../python/sglang/srt/layers/linear.py) | 矩阵投影 |
| [python/sglang/srt/layers/sampler.py](../../python/sglang/srt/layers/sampler.py) | 采样 |
| [python/sglang/srt/layers/attention/torch_native_backend.py](../../python/sglang/srt/layers/attention/torch_native_backend.py) | Attention 参考执行 |
