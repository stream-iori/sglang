# 04 模型：地址表怎样接到下一个 token

[返回全局地图](README.md) · 上一页：[KV 与工作单](02-memory-batch.md) · 下一页：[分支与回收](04-branches.md)

**Scheduler 决定算哪些输入、写哪些地址；模型把这些输入变成词表分数。** 本页解释 tiny runner 的 NumPy 单层 Transformer。前文的 10、11 是脚本示例，不是 tiny 模型的保证输出。

## 放大总图的节点 04

```text
input_ids
    |
Embedding：编号 -> hidden 向量
    |
RMSNorm -> Q / K / V 投影 -> RoPE 作用于 Q、K
    |                            |
    |                      新 K/V 写 out_cache_loc
    |                            |
    +------ Q 与历史 K/V 做 Attention <--- 地址表找到历史 slots
                             |
                    输出投影 + 残差
                             |
                    RMSNorm -> SwiGLU FFN + 残差
                             |
                    Final RMSNorm -> LM head
                             |
                     logits -> argmax -> token
                             |
                        回到节点 05
```

| 节点 | 大白话 | 输入输出关系 |
|---|---|---|
| Embedding | 查词表，把编号变成向量 | 一个 id → D 个特征 |
| RMSNorm | 调整一个 token 向量的尺度 | 形状不变 |
| Q/K/V | Q 用来查询，K 用来匹配，V 提供汇总内容 | 从 hidden 分别投影 |
| RoPE | 把位置关系加入 Q/K | 当前 position 不能因为命中 prefix 就重置为 0 |
| Attention | 按匹配权重汇总可见历史 V | 跨 token 交换上下文信息 |
| FFN / SwiGLU | 对每个 token 的特征再加工 | 不直接跨 token 混合 |
| 残差 | 把分支结果加回原输入 | 保留原表示的通路 |
| LM head | 给每个词表 token 打分 | D 个特征 → Vocab 个 logits |
| sampling | 根据分数选一个编号 | tiny 使用 argmax，选择最高分 |

这里 D 是 hidden size，Vocab 是词表大小；不要把 value 向量 V 和 Vocab 混为一谈。logits 还不是概率。

## 同一套模型，EXTEND 与 DECODE 只改变本轮输入范围

| 模式 | 新输入数 | 读取哪些 KV | 结果怎么用 |
|---|---|---|---|
| 完整 EXTEND | 多个未命中 token | 每个位置只看自己及之前的位置 | 最后有效位置产生首 token |
| 中间 chunk | 一段 token | 已有 prefix + 当前段可见部分 | 写 KV，尚不确认输出 |
| DECODE | 每请求一个 token | 全部历史 + 当前 token | 产生下一个 token |

KV cache 省掉历史 token 的 K/V 重算，**没有省掉当前 Q 读取历史 K/V 的 Attention**。tiny 实现按 position 逐个处理；不要把这当成真实 GPU prefill 的并行实现方式。

完整 prefix 命中时，tiny 另存的最后 hidden state 支持重新得到 logits；仅有 K/V 不能直接等同于最后输出分数。这是教学实现的补充设计，详见[完整命中](../model-execution-bridge.md#7-radix-完整命中为什么可能没有-input_ids)。

## 只在这里向下钻公式与算子

```text
先懂职责 -> 看 shape -> 推公式 -> 看 kernel 如何实现
概念文档    连接层      数学文档    Triton / CUDA 示例
```

自检：新输出 11 是否已经有 KV？没有；本轮输入 10 的 hidden 经过模型产生 11 的分数，尚未对 token 11 做 Embedding 和 K/V 投影。

深入：[Transformer 概念](../transformer-concept.md)、[数学](../transformer-math.md)、[Triton 对应算子](../triton-transformer.md)。源码：[`TinyTransformerModel`](../../src/my_sglang/tiny_transformer.py)。
