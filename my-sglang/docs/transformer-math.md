# Transformer 核心数学

本文集中保存 decoder-only Transformer 的核心公式、符号读法和 shape 变化。想先建立直觉、
暂时不看公式，可以阅读 [Transformer 基础概念](transformer-concept.md)；想看这些数学步骤
如何拆到教学 Triton kernel 中，可以阅读
[Triton 示例中的 Transformer 数据流](triton-transformer.md)。

本文采用“每个 token 是一个行向量”的写法。不同代码库可能把权重矩阵转置保存，但只要输入、
输出 shape 对得上，表达的是同一个线性变换。

## 1. 先读懂符号和 shape

| 符号 | 常见读法 | 含义 |
|---|---|---|
| $x\in\mathbb{R}^{D}$ | “x 属于 D 维实数空间” | 一个 token 的 hidden state，包含 $D$ 个实数。 |
| $X\in\mathbb{R}^{T\times D}$ | “X 属于 T 乘 D 维实数矩阵空间” | $T$ 个 token，每个 token 有 $D$ 个 hidden features。 |
| $W\in\mathbb{R}^{D\times K}$ | “W 属于 D 乘 K 维实数矩阵空间” | 把输入从 $D$ 维投影到 $K$ 维的权重。 |
| $xW$ | “x 乘 W” | 向量与矩阵相乘，混合输入 feature。 |
| $a\odot b$ | “a 与 b 逐元素相乘” | 相同 shape 的两个向量按位置相乘，不是矩阵乘法。 |
| $\sum_j a_j$ | “对所有 j 的 a-j 求和” | 沿下标 $j$ 所在的维度做归约。 |
| $W^\mathsf{T}$ | “W 的转置” | 交换矩阵的两个维度。 |
| $\phi(x)$ | “phi 作用于 x” | 对 $x$ 应用某个函数，常表示激活函数。 |

常用 shape 符号：

| 符号 | 含义 |
|---|---|
| $B$ | batch size，同时处理的序列数。 |
| $T$ | 本轮输入 token 数；在 decode 中常为 1。 |
| $D$ | 模型 hidden dimension。 |
| $H$ | Attention head 数。 |
| $d_h$ | 每个 head 的维度，常见关系是 $D=H d_h$。 |
| $N$ | 当前 query 可以看到的上下文长度。 |
| $D_{ff}$ | FFN 中间维度，通常大于 $D$。 |
| $V$ | vocabulary size，词表大小。 |

公式应从括号最内层开始读。例如 $\phi(xW_1+b_1)$ 的顺序是：先算 $xW_1$，再加
$b_1$，最后应用 $\phi$。

## 2. 一个 pre-norm Transformer block 的总公式

现代 decoder-only LLM 常采用 pre-norm block。把 Attention 和 FFN 的内部细节先折叠起来，
一个 block 可以写成：

$$
u = x + \operatorname{Attention}(\operatorname{RMSNorm}(x)),
$$

$$
y = u + \operatorname{FFN}(\operatorname{RMSNorm}(u)).
$$

第一式读作：“先归一化 $x$，送入 Attention，再把 Attention 的结果加回原来的 $x$，得到
$u$。”第二式同理：“先归一化 $u$，送入 FFN，再把 FFN 的结果加回 $u$，得到 block 输出
$y$。”

这两次加法就是 residual connection。它们要求子层输出与残差主路具有相同 shape：输入是
`[B,T,D]`，block 输出仍是 `[B,T,D]`。

下面按这条数据流依次展开每一个数学步骤。

<a id="math-rmsnorm"></a>

## 3. RMSNorm：先稳定每个 token 的整体尺度

对一个 token 的 hidden state $x\in\mathbb{R}^{D}$，先计算 mean square：

$$
m = \frac{1}{D}\sum_{j=1}^{D}x_j^2.
$$

读作：“$m$ 等于 $x$ 的每个元素平方后求和，再除以 hidden dimension $D$。”然后计算
inverse RMS：

$$
r = \frac{1}{\sqrt{m+\epsilon}}.
$$

读作：“$r$ 等于 $m$ 加 epsilon 后开平方，再取倒数。”最后对每个 feature 应用同一个
行级缩放 $r$，再乘可学习权重：

$$
y_i = x_i\,r\,\gamma_i.
$$

读作：“输出的第 $i$ 个元素，等于输入的第 $i$ 个元素乘整行共享的 $r$，再乘第 $i$ 个
可学习权重 $\gamma_i$。”其中 $\gamma\in\mathbb{R}^{D}$，代码中常叫 `weight`。

合在一起就是：

$$
\operatorname{RMSNorm}(x)_i
= \gamma_i\frac{x_i}{\sqrt{\frac{1}{D}\sum_{j=1}^{D}x_j^2+\epsilon}}.
$$

### epsilon 的数学作用

$\epsilon$ 是所有 token 行共享的一个很小的正常数，不是可学习参数。它加在 mean square
之后、开平方之前，作用不只是“避免写出除零异常”：

1. 如果整行 $x=0$，那么 $m=0$。没有 $\epsilon$ 时，$r=1/0=\infty$，后续
   $x_i r$ 会出现 $0\times\infty$，可能产生 NaN。
2. 因为 $m\ge 0$，分母至少是 $\sqrt{\epsilon}$，所以
   $r\le 1/\sqrt{\epsilon}$。这限制了极小输入被放大的最大倍数。
3. $\epsilon$ 越大，数值保护越强；但当 $m$ 与 $\epsilon$ 同量级时，$\epsilon$ 也会更明显地
   改变归一化结果。因此不能为了“更稳定”随意更改已有模型的配置。
4. $\epsilon=0$ 会失去保护；$\epsilon<0$ 还可能令 $m+\epsilon<0$，使平方根产生 NaN。

例如 $\epsilon=10^{-5}$ 时，$1/\sqrt{\epsilon}\approx316.2$；$\epsilon=10^{-6}$ 时上限约为
1000。这里限制的是 $r$，最终输出还会乘 $x_i$ 和 $\gamma_i$。

### 一个数值例子

忽略 $\epsilon$，并令 $\gamma=[1,1]$，输入 $x=[3,4]$：

```text
m = (3² + 4²) / 2 = 12.5
r = 1 / sqrt(12.5) ≈ 0.283
y = [3 × 0.283, 4 × 0.283] ≈ [0.85, 1.13]
```

输出的 RMS 约等于 1，但两个 feature 没有变成相同的数。统一缩放不会在乘 $\gamma$ 之前
改变元素的符号和相对比例。

### RMSNorm 与 LayerNorm

LayerNorm 还会先减去整行均值 $\mu$：

$$
\operatorname{LayerNorm}(x)_i
= \gamma_i
\frac{x_i-\mu}{\sqrt{\frac{1}{D}\sum_{j=1}^{D}(x_j-\mu)^2+\epsilon}}
+\beta_i.
$$

RMSNorm 没有 $x_i-\mu$ 这一步，也通常没有最后的偏移 $\beta_i$。所以两者不是同一个公式，
不能在已有模型权重中随意互换。

## 4. 线性投影与矩阵乘

Transformer 中的 Q/K/V、Attention 输出、FFN 和 LM head 都依赖线性投影。对一批 token：

$$
X\in\mathbb{R}^{T\times D},\qquad
W\in\mathbb{R}^{D\times K},\qquad
b\in\mathbb{R}^{K},
$$

$$
Y=XW+b,\qquad Y\in\mathbb{R}^{T\times K}.
$$

读作：“$X$ 乘权重 $W$，再沿 token 维广播加上 bias $b$，得到 $Y$。”每个输出元素为：

$$
Y_{tk}=\sum_{d=1}^{D}X_{td}W_{dk}+b_k.
$$

这里沿输入 feature 下标 $d$ 求和；不同 token 下标 $t$ 之间没有相加。因此线性层会混合一个
token 内部的 features，但不会单独完成 token 之间的信息交换。

有些 LLM 线性层不使用 bias，此时去掉 $b$ 即可。权重在代码中也可能以 `[K,D]` 保存并使用
$W^\mathsf{T}$；判断是否正确应看乘法两端的 shape，而不是只看变量命名。

一般矩阵乘可以写成：

$$
A\in\mathbb{R}^{M\times K},\quad
B\in\mathbb{R}^{K\times N},\quad
C=AB\in\mathbb{R}^{M\times N},
$$

$$
C_{mn}=\sum_{k=1}^{K}A_{mk}B_{kn}.
$$

中间维度 $K$ 必须匹配，并在结果中被求和消去。

<a id="math-qkv"></a>

## 5. Q、K、V：把同一份输入分成三种角色

设归一化后的 hidden states 为 $X_n\in\mathbb{R}^{T\times D}$：

$$
Q=X_nW_Q,\qquad K=X_nW_K,\qquad V=X_nW_V.
$$

读作：“同一个输入分别乘三组权重，得到 query、key 和 value。”它们的直观角色是：

- query：当前 token 想找什么信息。
- key：每个可见 token 提供什么匹配标签。
- value：匹配后真正取回什么内容。

投影结果通常会 reshape 成多个 heads。简化地写：

$$
Q,K,V\in\mathbb{R}^{T\times H\times d_h}.
$$

标准多头 Attention 常满足 $D=H d_h$，但 GQA/MQA 中 Q head 数与 K/V head 数可以不同。
这不会改变“query 与 key 算相关性、再用权重汇总 value”的核心数学。

<a id="math-rope"></a>

## 6. RoPE：把位置信息旋转进 Q 和 K

仅有 Q/K 内容向量时，Attention 不知道 token 的先后位置。RoPE（Rotary Position
Embedding）把相邻两个 feature 看成一个二维坐标，并按位置相关角度旋转。对某一对坐标：

$$
\begin{aligned}
x'_{2i} &= x_{2i}\cos\theta - x_{2i+1}\sin\theta,\\
x'_{2i+1} &= x_{2i}\sin\theta + x_{2i+1}\cos\theta.
\end{aligned}
$$

读作：“两个相邻 feature 按角度 $\theta$ 做二维旋转。”不同位置、不同 feature pair 使用的
角度不同。RoPE 通常应用于 Q 和 K，不直接应用于 V：

$$
\widetilde Q=\operatorname{RoPE}(Q,\text{position}),\qquad
\widetilde K=\operatorname{RoPE}(K,\text{position}).
$$

旋转保持每个二维 pair 的长度，但会改变不同位置的 Q/K 点积，从而让 Attention 感知相对
位置。具体频率和 scaling 方案由模型配置决定。

<a id="math-attention"></a>

## 7. Scaled dot-product Attention

对固定的 batch 和 head，当前 query 为 $q\in\mathbb{R}^{d_h}$，第 $n$ 个历史 key/value
分别为 $k_n,v_n\in\mathbb{R}^{d_h}$。

### 第一步：query 与每个 key 算相关性

$$
s_n=\frac{q\cdot k_n}{\sqrt{d_h}}.
$$

其中点积为：

$$
q\cdot k_n=\sum_{d=1}^{d_h}q_d k_{n,d}.
$$

读作：“把 query 与第 $n$ 个 key 的对应 feature 相乘后求和，再除以 head dimension 的
平方根。”$1/\sqrt{d_h}$ 防止维度增大时点积幅度随之过大，避免 Softmax 过早饱和。

<a id="math-causal-mask"></a>

### 第二步：加入 causal mask

decoder-only 模型不能读取未来 token。对位置 $t$ 的 query，可以定义：

$$
M_{tn}=
\begin{cases}
0,& n\le t,\\
-\infty,& n>t.
\end{cases}
$$

实际送入 Softmax 的分数是 $s_{tn}+M_{tn}$。未来位置加上 $-\infty$ 后，其指数为 0，最终
Attention 权重也为 0。decode 阶段常直接只传入可见的历史 K/V，此时 causal 条件已经由输入
范围保证，不一定需要在 kernel 中再物化一张 mask。

<a id="math-softmax"></a>

### 第三步：Softmax 变成权重

对一行 $N$ 个分数：

$$
p_n=\frac{e^{s_n}}{\sum_{j=1}^{N}e^{s_j}}.
$$

所有 $p_n$ 非负且总和为 1。为避免 $e^{s_n}$ 溢出，实际采用稳定形式：

$$
m=\max_j s_j,\qquad
p_n=\frac{e^{s_n-m}}{\sum_{j=1}^{N}e^{s_j-m}}.
$$

给整行同时减去同一个常数不会改变 Softmax 结果，但会让最大的指数变成 $e^0=1$，显著降低
数值溢出风险。

### 第四步：用权重汇总 V

$$
o=\sum_{n=1}^{N}p_n v_n.
$$

读作：“用每个位置的 Attention 权重 $p_n$ 乘对应 value，再沿上下文维求和。”输出 $o$ 的
shape 仍是 `[d_h]`。所以 Attention 的本质不是返回某一个历史 token，而是返回历史 values
的加权组合。

<a id="math-online-softmax"></a>

### online Softmax 为什么与完整 Softmax 等价

如果上下文很长，不必先保存全部 $N$ 个 scores。可以逐 tile 处理，并维护三个状态：

| 状态 | 含义 |
|---|---|
| $m$ | 已处理 scores 的最大值。 |
| $l$ | 在共同指数基准 $m$ 下的指数和。 |
| $a$ | 在共同基准 $m$ 下的加权 V 和。 |

旧状态 $(m,l,a)$ 与当前 tile 合并时，先更新最大值：

$$
m'=\max(m,\max(s_{tile})).
$$

然后把旧状态重标定到新的指数基准：

$$
\alpha=e^{m-m'},
$$

$$
l'=l\alpha+\sum_{n\in tile}e^{s_n-m'},
$$

$$
a'=a\alpha+\sum_{n\in tile}e^{s_n-m'}v_n.
$$

处理完所有 tile 后：

$$
o=\frac{a}{l}.
$$

最大值变化时，旧的分母与旧的加权和必须同时乘 $\alpha$；否则不同 tile 会使用不同的指数
基准，结果就不再等价于完整 Softmax。

<a id="math-attention-output"></a>

## 8. 多头合并、输出投影和第一条残差

每个 head 独立得到 $o_h\in\mathbb{R}^{d_h}$。先拼接所有 heads：

$$
o_{cat}=\operatorname{Concat}(o_1,\ldots,o_H)\in\mathbb{R}^{D},
$$

再做输出投影：

$$
a=o_{cat}W_O,\qquad W_O\in\mathbb{R}^{D\times D}.
$$

输出投影把多个 heads 的结果重新混合回模型 hidden dimension。随后做残差相加：

$$
u=x+a.
$$

这里 $x$ 和 $a$ 都必须是 `[D]`。残差是逐元素相加，不是把两个向量拼接起来。

<a id="math-ffn"></a>

## 9. FFN、SiLU 与 SwiGLU

Attention 负责在 token 之间汇总信息；FFN 对每个 token 独立混合 hidden features。

<a id="math-classic-ffn"></a>

### 经典两层 FFN

$$
h=\phi(xW_1+b_1),
$$

$$
y=hW_2+b_2.
$$

第一层把 $D$ 维升到 $D_{ff}$，激活函数 $\phi$ 引入非线性；第二层再降回 $D$，以便与残差
主路相加。如果去掉中间非线性，两个线性层可以合并成一个线性变换，表达能力会受限。

### SiLU

$$
\operatorname{SiLU}(z)
=z\,\sigma(z)
=\frac{z}{1+e^{-z}}.
$$

读作：“SiLU of z 等于 z 乘 z 的 sigmoid。”如果先做 bias add，则令 $z_i=x_i+b_i$，再
逐元素应用 SiLU。

<a id="math-swiglu"></a>

### SwiGLU 门控 FFN

现代 LLM 常使用两条输入投影：

$$
g=\operatorname{SiLU}(xW_{gate}),
$$

$$
u=xW_{up},
$$

$$
y=(g\odot u)W_{down}.
$$

读作：“gate 分支经过 SiLU，与 up 分支逐元素相乘，再通过 down projection 回到 hidden
dimension。”常见 shapes 为：

```text
x             [D]
g, u          [D_ff]
g ⊙ u         [D_ff]
y             [D]
```

门控乘法 $\odot$ 不沿任何维度求和；只有三个 projection 中的矩阵乘会归约输入 feature 维。
完成 FFN 后，再与进入该子层前保留的 residual hidden state 相加，得到 block 输出。这里不要
把 residual hidden state 与上式中表示 up 分支的 $u$ 混为一谈。

<a id="math-lm-head"></a>

## 10. Final RMSNorm 与 LM head

经过 $L$ 个 blocks 后，模型通常再做一次 Final RMSNorm：

$$
z=\operatorname{RMSNorm}(x_L).
$$

然后 LM head 把 hidden dimension 映射到词表：

$$
W_{vocab}\in\mathbb{R}^{V\times D},\qquad
\operatorname{logits}=zW_{vocab}^{\mathsf T}\in\mathbb{R}^{V}.
$$

每个 logit 对应一个候选 token 的未归一化分数。是否继续做 temperature、top-k、top-p 或
argmax 属于 sampling 阶段，不是 Transformer block 本身。

<a id="math-sampling"></a>

## 11. 从 logits 到 token：sampling

设词表中第 $i$ 个 token 的 logit 是 $l_i$。**greedy sampling** 不引入随机性，直接选择
分数最大者：

$$
\operatorname{next\_token}=\arg\max_i l_i.
$$

temperature $\tau>0$ 会在 Softmax 前缩放 logits：

$$
p_i=\frac{\exp(l_i/\tau)}{\sum_j\exp(l_j/\tau)}.
$$

- $\tau<1$：放大分数差异，概率更集中；
- $\tau>1$：缩小分数差异，概率更平坦；
- greedy 直接比较 logits，不需要先算 Softmax。

**top-k** 只保留 logits 最大的 $k$ 个候选，把其余候选视为 $-\infty$，再在保留集合中
归一化和采样。**top-p** 先按概率从大到小排序，保留累计概率第一次达到阈值 $p$ 的最小
候选集合，再重新归一化和采样。

`TinyTransformerRunner` 只实现 greedy，使相同输入和固定权重始终得到相同 token，便于测试
Scheduler、KV slot 和请求生命周期。真实 serving runner 还要根据每个请求的 sampling
parameters 分别处理 temperature、top-k、top-p、随机数状态和约束解码。

## 12. 按数据流做 shape 检查

忽略 batch 维，对一批 $T$ 个 token，常见 shape 流为：

```text
hidden states                    [T, D]
RMSNorm                          [T, D]
Q/K/V projection                [T, H, d_h]
Attention per-head output        [T, H, d_h]
concat + output projection       [T, D]
first residual                   [T, D]
RMSNorm                          [T, D]
gate/up projection               [T, D_ff]
elementwise gate                 [T, D_ff]
down projection                  [T, D]
second residual                  [T, D]
Final RMSNorm                    [T, D]
LM head logits                   [T, V]
```

检查公式或代码时，可以依次问：矩阵乘的中间维是否匹配、归约发生在哪一维、逐元素运算的
shape 是否相同或可广播、残差两侧 shape 是否完全一致。只要这四类问题能回答清楚，大多数
Transformer 数学和 kernel shape 错误都能快速定位。
