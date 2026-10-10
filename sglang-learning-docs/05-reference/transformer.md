# Transformer 的一轮推理与 SGLang 对应关系

> 源码基准：learning `2aa70e3eb4`，已合入 origin/main `6fc8d9da32`（2026-10-10）。

一个 token ID 经 embedding 变成向量，多层 Transformer 用上下文更新表示，最后给词表里每个候选 token 打分。

```text
input_ids ─ embedding ─ hidden states
                          │
              ┌───────────▼────────────┐
              │ Norm → Q/K/V → RoPE    │
              │      → Attention      │ × L 层
              │ 残差 → Norm → MLP     │
              │      → 残差           │
              └───────────┬────────────┘
                          ▼
               final norm → lm_head → logits → sampler
```

| 概念 | 用途 | SGLang 所在层 |
|---|---|---|
| embedding | ID 查向量 | 模型实现 |
| Q/K/V | query 对历史 key 打分并聚合 value | Attention 层/backend |
| RoPE | 在 Q/K 中加入位置关系 | 模型/算子 |
| KV cache | 保留历史 K/V，避免下轮重算 | pool + cache + 地址映射 |
| MLP | 对每个位置的特征做非线性变换 | 模型/linear/activation |
| logits | 词表候选分数 | lm_head / logits processor |
| sampling | 从分数得到下一个 ID | sampler |

## Prefill 和 decode 为什么计算量不同

| 路径 | 新 query 数 | 历史 KV |
|---|---|---|
| prefill/extend | 本轮新增输入位置数 | 可包括复用前缀 |
| 普通 decode | 每请求通常 1 | 随上下文增长 |

decode 不是重新计算整个 prompt，也不是只算一层。它执行完整模型层，用本轮输入位置的 Q/K/V 加上历史 KV 得到新分数。

## Q heads 与 KV heads

Qwen3-0.6B：Q heads=16，KV heads=8，head_dim=128。多个 query heads 可共享 KV heads（GQA）；每 token KV 字节应使用 KV head 数。

head_dim 不一定等于 hidden_size / Q heads。这个本地配置 hidden_size=1024，16×128=2048，模型投影维度可不同于残差维度，应以配置和实现为准。

缓存字节计算见 [性能直觉](performance-intuition.md)，矩阵并行见 [多卡](../03-advanced/multi-gpu.md)。

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/models/qwen3.py](../../python/sglang/srt/models/qwen3.py) | Qwen3 实际结构 |
| [python/sglang/srt/layers/attention/torch_native_backend.py](../../python/sglang/srt/layers/attention/torch_native_backend.py) | Attention 计算 |
| [python/sglang/srt/layers/logits_processor.py](../../python/sglang/srt/layers/logits_processor.py) | logits 生成 |
