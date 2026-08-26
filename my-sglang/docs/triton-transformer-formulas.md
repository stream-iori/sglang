# Triton 示例中的 Transformer 公式

本文集中解释 `examples/triton/02`～`07` 用到的数学公式、张量形状及其在
Transformer 中的作用。重点是先回答“为什么算这个”，再把公式中的符号对应到 Triton
变量；CUDA/Triton 的 program、tile、logical lane 等执行概念见
[Triton 与 CUDA：代码概念映射](triton-cuda-basics.md)。FFN、MLP 和门控 MLP 等术语的
区别见 [Transformer 基础概念](transformer-concept.md)。

## 1. 从 Transformer 数据流看这些示例

```text
hidden states
     │
     ├─ RMSNorm ──> Q/K/V 线性投影（矩阵乘）
     │                         │
     │                         └─ scaled dot-product attention（含 softmax）
     │                                              │
     └──────────────────────── residual <───────────┘
                                                    │
                           RMSNorm ──> FFN 矩阵乘 ──> SiLU/门控 ──> 矩阵乘
```

这些课程不是一个完整 Transformer block，而是从易到难拆出其中的计算原语：

| 示例 | 数学原语 | 在 Transformer 中的常见作用 |
|---|---|---|
| `02_fused_elementwise.py` | bias add、SiLU | FFN 的非线性激活；示例同时演示算子融合。 |
| `03_row_softmax.py` | 按行 Softmax | 把 Attention score 变成总和为 1 的权重。 |
| `04_rmsnorm.py` | RMSNorm | 在 Attention 或 FFN 前后稳定隐藏状态的尺度。 |
| `05_matmul.py` | 分块矩阵乘 | Q/K/V、Attention 输出以及 FFN 的线性投影。 |
| `06_autotune_matmul.py` | 同一个矩阵乘 | 说明公式相同时，不同 tile 仍会有不同 GPU 性能。 |
| `07_attention.py` | 单 query Attention、online Softmax | 用 Q 与历史 K/V 计算一次 decode Attention。 |

`01_vector_add.py` 只建立 Triton 索引和访存模型，不对应特定 Transformer 公式。

## 2. SiLU：给线性网络加入非线性

令逐元素相加结果为

$$
z_i = x_i + b_i,
$$

则 SiLU（Sigmoid Linear Unit）为

$$
\operatorname{SiLU}(z_i)
= z_i\,\sigma(z_i)
= \frac{z_i}{1 + e^{-z_i}}.
$$

如果只有线性层而没有激活函数，多层线性变换仍可合并成一个线性变换。SiLU 提供非线性，
通常出现在 Transformer 的 FFN 中；现代模型也常在 SwiGLU 等门控 FFN 中使用 SiLU。

示例 02 为了专注一维 fusion，要求 `x` 和 `bias` 形状相同。真实线性层的 bias 常沿 token
维广播，SwiGLU 还会再乘一个 gate 分支，因此示例并不是完整 FFN。

代码对应关系：

| 公式 | Triton 代码 | 数据所在位置 |
|---|---|---|
| $z_i=x_i+b_i$ | `z = x + bias` | 当前 program 的 tile 内，不写回 HBM。 |
| $\sigma(z_i)$ | `tl.sigmoid(z)` | 当前 program 内。 |
| $z_i\sigma(z_i)$ | `output = z * tl.sigmoid(z)` | 最后一次 `tl.store` 才写回 HBM。 |

融合的主要目的不是改变公式，而是避免把中间数组 $z$ 写到 HBM 后又读回来。

## 3. Softmax：把一行分数变成概率权重

对一行长度为 $N$ 的分数 $x$，Softmax 定义为

$$
p_i = \frac{e^{x_i}}{\sum_{j=1}^{N} e^{x_j}}.
$$

直接计算 $e^{x_i}$ 可能溢出。因为给整行同时减去常数不会改变结果，实际采用稳定形式：

$$
m = \max_j x_j, \qquad
p_i = \frac{e^{x_i-m}}{\sum_{j=1}^{N} e^{x_j-m}}.
$$

这对应示例 03 的 `tl.max`、`tl.exp`、`tl.sum` 和除法。Attention 会对每个 query 的一行
score 做 Softmax，使权重非负且总和为 1，再用它们加权 V。

`BLOCK_SIZE` 取不小于 $N$ 的最小二次幂，因此最后一些 logical lanes 只是 padding。加载
padding 时使用 $-\infty$：

$$
e^{-\infty}=0,
$$

所以无效列既不会抬高最大值，也不会进入分母；`tl.store` 的 mask 则阻止它们写出界。

## 4. RMSNorm：稳定隐藏状态的尺度

对于隐藏维度为 $D$ 的一行 $x$，RMSNorm 为

$$
\operatorname{RMS}(x)
= \sqrt{\frac{1}{D}\sum_{j=1}^{D}x_j^2 + \epsilon},
$$

$$
y_i = \gamma_i\frac{x_i}{\operatorname{RMS}(x)}.
$$

其中 $\gamma_i$ 是可学习权重，$\epsilon$ 防止分母为零。它把一行 hidden state 的均方根
尺度归一化，使深层网络训练和推理中的数值范围更稳定。

RMSNorm 与 LayerNorm 的关键区别是它不减均值：RMSNorm 没有
$x_i-\operatorname{mean}(x)$ 这一步。示例 04 让一个 Triton program 负责一整行，从而在
同一个 program 内完成平方和归约、倒平方根和逐元素权重乘法。

| 数学量 | Triton 变量 |
|---|---|
| $x_i$ | `x_fp32` |
| $D^{-1}\sum_jx_j^2$ | `mean_square` |
| $1/\sqrt{\operatorname{mean_square}+\epsilon}$ | `inv_rms` |
| $\gamma_i$ | `weight` |
| $y_i$ | `output` |

平方和使用 FP32，是因为归约会累加许多项；即使输入以后换成 FP16/BF16，FP32 累加通常也
能减小舍入误差。padding lane 加载 0，因而不会增加平方和；分母仍除以真实维度 `n_cols`，
而不是 `BLOCK_SIZE`。

## 5. 矩阵乘：Transformer 线性层的主体

设

$$
A\in\mathbb{R}^{M\times K},\qquad
B\in\mathbb{R}^{K\times N},
$$

矩阵乘输出

$$
C=AB,\qquad
C_{mn}=\sum_{k=1}^{K}A_{mk}B_{kn}.
$$

在线性层 $Y=XW+b$ 中，可以把 $M$ 理解为本轮处理的 token 数，$K$ 是输入 hidden
dimension，$N$ 是输出 dimension。Q/K/V 投影、Attention 输出投影和 FFN 的升维/降维都以
这种矩阵乘为主体；示例 05 没有融合 bias。

一个 program 只负责输出矩阵的一块：

```text
C 的一个 [BLOCK_M, BLOCK_N] tile
       = A 的 [BLOCK_M, K] 横条
       × B 的 [K, BLOCK_N] 竖条
```

由于不能一次加载完整的 $K$ 维，kernel 再按 `BLOCK_SIZE_K` 分段：

$$
C_{tile}
= \sum_{t} A_{tile,t}B_{t,tile}.
$$

`tl.dot(a, b, accumulator)` 对当前 K tile 做乘加，`accumulator` 跨循环保存部分和。M、N
或 K 不能整除 tile 时，越界输入以 0 加载，不改变点积；越界输出由 store mask 丢弃。

示例 06 的数学结果完全相同。Autotune 改变的是 `BLOCK_SIZE_M/N/K` 和 `num_warps`，也就是
一个 program 做多少工作以及如何映射到底层 GPU 资源。公式相同不代表不同配置的访存、
并行度、寄存器压力和最终耗时相同。

## 6. Scaled dot-product Attention

示例 07 处理 decode 阶段的单 query。其形状为：

$$
Q\in\mathbb{R}^{B\times H\times D},\qquad
K,V\in\mathbb{R}^{B\times H\times N\times D},
$$

其中 $B$ 是 batch size，$H$ 是 head 数，$N$ 是可见上下文长度，$D$ 是 head dimension。
对固定的 `(batch, head)`，公式为

$$
s_n = \frac{q\cdot k_n}{\sqrt D},
\qquad
p_n = \frac{e^{s_n}}{\sum_{j=1}^{N}e^{s_j}},
\qquad
o = \sum_{n=1}^{N}p_nv_n.
$$

三个步骤的作用分别是：

1. $q\cdot k_n$ 衡量当前 query 与第 $n$ 个 key 的相关性。
2. $1/\sqrt D$ 避免维度增大时点积幅度过大，防止 Softmax 过早饱和。
3. Softmax 权重对 V 加权求和，得到当前 head 的上下文表示。

示例没有显式 causal mask。它假设传入的 K/V 已经只包含当前 query 允许看到的上下文；若
传入未来 token，它也会参与 Attention。示例也没有 GQA、paged KV cache 或变长 batch，
不能直接替代生产级 Attention kernel。

### 为什么需要 online Softmax

普通实现会先生成完整的 $N$ 个 score，再做 Softmax。示例 07 每次只处理 `BLOCK_N` 个
上下文位置，需要在跨 tile 时保留三个状态：

| 状态 | 数学含义 | Triton 变量 |
|---|---|---|
| $m$ | 已处理 score 的最大值 | `running_max` |
| $l$ | 在共同基准 $m$ 下的指数和 | `running_sum` |
| $a$ | 在共同基准 $m$ 下的加权 V 和 | `accumulator` |

旧状态 $(m,l,a)$ 与当前 tile 合并时，先求

$$
m' = \max(m,\max(s_{tile})),
$$

再把旧状态从基准 $m$ 重标定到新基准 $m'$：

$$
\alpha=e^{m-m'},
$$

$$
l' = l\alpha + \sum_{n\in tile}e^{s_n-m'},
$$

$$
a' = a\alpha + \sum_{n\in tile}e^{s_n-m'}v_n.
$$

处理完所有 tile 后：

$$
o=\frac{a}{l}.
$$

`old_scale` 就是 $\alpha$。最大值变大时，旧的分母与加权和必须同时缩放，否则不同 tile
会处在不同的指数基准上，最终结果不再等价于完整 Softmax。

## 7. 阅读代码时的形状检查

遇到二维 pointer 表达式时，可以先忽略具体地址，写出每个逻辑张量的 shape：

| 示例 | 当前 program 内的主要值 | shape |
|---|---|---|
| 02 | `offsets`, `x`, `bias`, `output` | `[BLOCK_SIZE]` |
| 03 | `row`, `softmax` | `[BLOCK_SIZE]` |
| 04 | `x_fp32`, `weight`, `output` | `[BLOCK_SIZE]` |
| 05/06 | `a`, `b`, `accumulator` | `[BM,BK]`, `[BK,BN]`, `[BM,BN]` |
| 07 | `query`, `keys`, `scores`, `values`, `accumulator` | `[BD]`, `[BN,BD]`, `[BN]`, `[BN,BD]`, `[BD]` |

再依次检查：grid 选择了哪个输出块、offset 是否覆盖目标 shape、mask 是否只保留真实元素、
reduction 在哪一维发生、最终 store 是否与输出 shape 一致。这样可以把 Transformer 数学和
Triton 地址计算分开理解，再验证二者是否正确连接。
