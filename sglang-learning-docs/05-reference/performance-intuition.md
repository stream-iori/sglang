# 性能直觉：定量理解 LLM Serving 的设计决策

> 目标：用简单的数学建立对 LLM Serving 性能的定量直觉，理解 SGLang 每个设计决策背后的"数字原因"。
> 前置：[Week 2](../02-core-systems/scheduler-and-cache.md) 完成即可阅读。无需 GPU 经验。
> 时间：~2 小时
>
> **关联文档**:
> - Prefill/Decode 在代码中的体现 → [model-execution.md: EXTEND vs DECODE](../02-core-systems/model-execution.md)
> - KV Cache 内存池的代码实现 → [scheduler-and-cache.md: 内存池管理](../02-core-systems/scheduler-and-cache.md)
> - Continuous Batching 可运行 demo → [exercises.md 实验 1](../04-practice/exercises.md)
> - 概念混淆澄清 (ScheduleBatch/ForwardBatch) → [faq.md](../05-reference/faq.md)

---

## 一、GPU 硬件基础：两个关键数字

理解 LLM Serving 性能，只需要记住 GPU 的两个核心指标：

| 指标 | A100 (80GB) | H100 | 大白话 |
|---|---|---|---|
| **算力 (FLOPS)** | 312 TFLOPS (BF16) | 990 TFLOPS (BF16) | 每秒能做多少次乘加 |
| **显存带宽** | 2.0 TB/s | 3.35 TB/s | 每秒能搬多少数据 |

**关键比值: 算术强度 (Arithmetic Intensity)**

```
算力 / 带宽 = 312T / 2T = 156 FLOPS/Byte (A100)
```

这意味着：**GPU 每从显存读 1 Byte 数据，能做 156 次浮点运算**。

如果一个操作需要读很多数据但计算很少（比值 < 156），GPU 就会"等数据"而不是"等计算"。这就是 **memory-bound**。

---

<a id="perf-prefill-vs-decode"></a>

## 二、Prefill vs Decode：为什么性质完全不同

### Prefill (EXTEND): 计算密集

```
输入: prompt = N 个 token
操作: Self-Attention = Q × K^T (矩阵乘法)
  Q: shape [N, d]
  K: shape [N, d]
  计算量: 2 × N × N × d FLOPS
  数据量: 2 × N × d × 2 Bytes (BF16)

算术强度 = (2 × N × N × d) / (2 × N × d × 2) = N / 2
```

**当 N = 1024 时，算术强度 = 512 > 156 → Compute-bound!**

GPU 的计算单元跑满，带宽还有余裕。此时 **增大 batch size 不会显著提升吞吐**（计算已满载）。

### Decode: 内存密集

```
输入: 1 个新 token (query)
操作: 这 1 个 query 要和所有历史 KV 做 attention
  Q: shape [1, d]
  K: shape [S, d]  (S = 历史序列长度)
  计算量: 2 × 1 × S × d FLOPS
  数据量: S × d × 2 Bytes (读取整个 KV Cache)

算术强度 = (2 × S × d) / (S × d × 2) = 1
```

**算术强度 = 1 << 156 → Memory-bound!**

GPU 绝大部分时间在"等数据从显存搬过来"，计算单元严重闲置。

### 数字对比

以 Llama-3-8B (d=4096, 32 layers) 为例，处理一个请求 (prompt=512, generate=100):

| 阶段 | 计算量 | 数据搬运量 | GPU 利用率 |
|---|---|---|---|
| Prefill 512 tokens | ~17 TFLOPS | ~67 MB | **高** (~55%) |
| Decode 1 token | ~34 GFLOPS | ~67 MB | **极低** (~0.1%) |

**关键洞察**: Decode 阶段每个 token 的计算量只有 Prefill 的 1/512，但要读的 KV Cache 数据量相同！

---

## 三、为什么 Batching 能拯救 Decode

Decode 阶段 GPU 利用率极低的原因是：每次只处理 1 个 token 的 attention，但要读取整个 KV Cache。

**如果同时处理 B 个请求呢？**

```
Batch Decode:
  Q: shape [B, d]  (B 个请求各出 1 个 query)
  每个请求读自己的 KV Cache

  总计算量: B × 2 × S × d
  总数据量: B × S × d × 2  (每个请求的 KV Cache 独立)

  算术强度 = 1 (不变! 因为计算和数据都 ×B)
```

等等，算术强度没变？那 batching 为什么有用？

**答案在模型权重**:

```
实际的 forward 不只是 attention，还有 FFN/MLP:
  权重: W shape [d, 4d] — 所有请求共享！
  单请求: x @ W → 计算 = 2×d×4d, 读 = d×4d×2 → 强度 = 1
  B 请求:  X @ W → 计算 = 2×B×d×4d, 读 = d×4d×2 → 强度 = B !!
```

**模型权重只需读一次，但可以同时为 B 个请求计算**。Batch size 越大，权重搬运的开销被摊薄得越多。

```
理论最优 batch size = 156 (A100)
  此时 MLP 部分刚好计算和带宽平衡
  实际受 KV Cache 内存限制，通常 batch 32~256
```

### Continuous Batching 的价值

Static Batching:
```
| req1 ████████████████ done |
| req2 ████████ done         |  ← req2 完成后 GPU 空转
| req3 ████████████ done     |
                      ↑ 浪费!
```

Continuous Batching:
```
| req1 ████████████████ done |
| req2 ████████ done → req4 █████ |  ← req2 完成立即填入 req4
| req3 ████████████ done → req5 ██|
                      ↑ GPU 永远满载!
```

**量化收益**: 假设请求生成长度服从均匀分布 [10, 100]，Continuous Batching 相比 Static Batching 吞吐提升约 **2-4x**。

---

## 四、KV Cache 内存占用计算

### 公式

```
KV Cache per token per layer = 2 × num_kv_heads × head_dim × dtype_bytes
                                ↑                              ↑
                              K 和 V 各一份                   BF16 = 2 bytes

KV Cache per token (全部层) = num_layers × 2 × num_kv_heads × head_dim × dtype_bytes
```

### 具体计算: Llama-3-8B

```
参数:
  num_layers = 32
  num_kv_heads = 8 (GQA, 不是 32)
  head_dim = 128
  dtype = BF16 (2 bytes)

每 token KV Cache = 32 × 2 × 8 × 128 × 2 = 131,072 bytes = 128 KB

一个请求 (seq_len=2048):
  2048 × 128 KB = 256 MB

一个请求 (seq_len=8192):
  8192 × 128 KB = 1 GB !!
```

### Napkin Math: 能服务多少并发请求？

```
A100 80GB:
  模型权重: ~16 GB (8B × BF16)
  剩余显存: ~64 GB
  
  seq_len=2048: 64GB / 256MB = 250 并发请求
  seq_len=8192: 64GB / 1GB = 64 并发请求
  seq_len=32K:  64GB / 4GB = 16 并发请求 !!
```

**关键洞察**: 序列越长，能同时服务的请求越少，batch size 被迫减小，GPU 利用率下降。这就是为什么长上下文模型的 serving 成本很高。

### 为什么 RadixCache 如此重要

```
场景: 100 个请求共享 1000 token 的 system prompt

无缓存: 100 × 1000 × 128KB = 12.5 GB (重复存储)
有缓存: 1 × 1000 × 128KB = 125 MB (共享存储)

节省: 12.375 GB → 可以多服务 ~48 个并发请求!
```

---

## 五、端到端延迟分析

### TTFT (Time To First Token) — 首 token 延迟

```
TTFT ≈ Prefill 时间 = prompt_tokens × FLOPS_per_token / GPU_FLOPS

Llama-3-8B, prompt=1024 tokens:
  每 token 计算量 ≈ 2 × 8B = 16 GFLOPS (近似: 2 × 参数量)
  Prefill 计算量 = 1024 × 16G = 16.4 TFLOPS
  A100 峰值 = 312 TFLOPS, 实际利用率 ~55%
  TTFT ≈ 16.4T / (312T × 0.55) ≈ 95ms
```

### TPS (Tokens Per Second) — 生成速度

```
单请求 decode:
  每 token 计算量 ≈ 16 GFLOPS
  但瓶颈是带宽: 需要读取模型权重 = 16 GB
  A100 带宽 = 2 TB/s
  每 token 时间 ≈ 16GB / 2TB/s = 8ms
  TPS ≈ 125 tokens/s (单请求)

Batch=32:
  权重只读一次 (16GB), 但为 32 个请求计算
  每 token 时间 ≈ max(16GB/2TB/s, 32×16G/312T) = max(8ms, 1.6ms) = 8ms
  总 TPS = 32 × 125 = 4000 tokens/s (32 个请求共享)
  每请求 TPS 不变 = 125 tokens/s
```

**关键洞察**: 增大 batch size 可以提升总吞吐（所有请求加起来），但单个请求的生成速度不会变快。

---

## 六、设计决策的定量理由

| 设计 | 定量理由 |
|---|---|
| **Prefill/Decode 分离调度** | Prefill compute-bound (利用率55%), Decode memory-bound (利用率0.1%)。混在一起会互相干扰 |
| **Continuous Batching** | Decode batch=1 利用率0.1%, batch=64 利用率6.4%。必须尽量塞满 batch |
| **RadixCache 前缀复用** | 共享 prompt 节省的不只是计算，更是 KV 内存。释放内存 → 更大 batch → 更高吞吐 |
| **Chunked Prefill** | 一次 prefill 1024 tokens 耗时 ~95ms，会 block decode batch。切成 256 chunk → 每 chunk 24ms，decode 延迟更稳定 |
| **PD 分离** | Prefill 抢 GPU 算力影响 decode 延迟。物理隔离后 decode 的 P99 延迟从 ~50ms 降到 ~10ms |
| **投机解码** | Decode 每 step 8ms 但 GPU 利用率极低。Draft 猜 5 个 token + target 一次验证 = 1 step 但产出 3-4 tokens。等效 TPS ×3 |

---

## 七、自测练习

### 练习 1: 计算 KV Cache

Llama-3-70B 参数:
- num_layers = 80
- num_kv_heads = 8 (GQA)
- head_dim = 128
- dtype = BF16

问题:
1. 每 token 的 KV Cache 大小？
2. seq_len=4096 时一个请求占多少显存？
3. 4×A100 (320GB) 能服务多少并发请求？（模型权重 ~140GB）

### 练习 2: 估算吞吐

某服务使用 Llama-3-8B on A100:
- 平均 prompt_len = 500
- 平均 generation_len = 200
- 目标 TTFT < 200ms

问题:
1. 单 A100 能满足 TTFT 要求吗？
2. 要达到 10000 output tokens/s 的总吞吐，最少需要多大 batch size？
3. 该 batch size 需要多少 KV Cache 显存？能装下吗？

### 练习 3: RadixCache 收益估算

场景: 客服聊天系统
- System prompt: 800 tokens (所有请求相同)
- 历史对话: 平均 2000 tokens (每个请求不同)
- QPS: 50 请求/秒

问题:
1. 无缓存时每秒需要 prefill 多少 tokens？
2. 有 RadixCache 时每秒需要 prefill 多少 tokens？
3. 节省了多少 prefill 计算？

---

## 参考答案

<details>
<summary>练习 1 答案</summary>

```
每 token = 80 × 2 × 8 × 128 × 2 = 327,680 bytes = 320 KB
seq_len=4096: 4096 × 320KB = 1.28 GB
可用显存: 320 - 140 = 180 GB
并发请求: 180GB / 1.28GB ≈ 140 个
```
</details>

<details>
<summary>练习 2 答案</summary>

```
1. TTFT = 500 × 16G / (312T × 0.55) = 46ms < 200ms ✓ 满足
2. 单请求 TPS ≈ 125 tokens/s
   需要 batch = 10000 / 125 = 80
3. KV Cache = 80 × (500+200) × 128KB = 7.2 GB
   总显存 = 16(模型) + 7.2(KV) = 23.2 GB → A100 80GB 轻松装下
```
</details>

<details>
<summary>练习 3 答案</summary>

```
1. 无缓存: 50 × (800 + 2000) = 140,000 tokens/s prefill
2. 有缓存: 50 × 2000 = 100,000 tokens/s (system prompt 缓存命中)
   首次请求仍需 800 prefill，之后全部复用
3. 节省: 50 × 800 = 40,000 tokens/s → 节省 28.6% prefill 计算
   如果同一用户多轮对话，节省更多 (历史对话也可缓存)
```
</details>
