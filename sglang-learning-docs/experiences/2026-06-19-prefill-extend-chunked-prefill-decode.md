# Prefill / Extend / Chunked Prefill / Decode

> **记录日期**: 2026-06-19
> **触发场景**: 看 MLX overlap 日志时,同时看到 `forward_mode=extend`、`prefill_rids`、`decode`、`chunked prefill continuation`,容易混淆这些词的边界。
> **一句话结论**: `extend` 是调度层的“给序列追加一段 token”;`prefill` 是模型执行层的“为 prompt/token 段建立 KV cache”;`chunked prefill` 是长 prompt 被拆成多段后多次 `extend`;`decode` 是每次用上一个 token 生成下一个 token。

---

## 先看总图

```text
一次请求的典型生命周期:

用户 prompt
   │
   ▼
prefill / extend 阶段
   │  处理 prompt token,写 KV cache,产出第一个生成 token
   │
   ▼
decode 阶段
   │  每轮处理 1 个 token,生成下一个 token
   │
   ▼
结束: 达到 max_new_tokens / EOS / stop 条件
```

从调度器视角看:

```text
extend
├── 新请求第一次进来: 内部走 prefill
├── prefix cache 命中: 只 extend 没命中的后缀
├── prompt 太长: 多次 extend,这就是 chunked prefill
└── mixed batch: 一个 batch 里可能有人 extend,有人 decode

decode
└── 生成阶段,每轮只处理上一步生成的 1 个 token
```

---

## 词义拆开

| 词 | 站在哪一层说 | 核心含义 | 一次处理几个 token |
|----|--------------|----------|--------------------|
| `prefill` | 模型执行层 | 处理 prompt/token 段,写 KV cache,产出第一个生成 token | 通常多个 |
| `extend` | Scheduler 调度层 | 给请求序列继续追加一段 token | 一个或多个 |
| `chunked prefill` | 调度策略 | prompt 太长,拆成多块,多次 extend 完成 prefill | 每块多个 |
| `decode` | 模型执行层 / 调度层 | 用上一步 token 生成下一 token | 通常 1 个 |

> 最容易混淆的是 `prefill` 和 `extend`:很多时候它们描述的是同一段动作,但角度不同。调度器说“我在 extend 这个请求”,模型 runner 说“这个新请求内部要 prefill”。

---

## 例子一:没有 cache 的普通请求

假设用户 prompt 被 tokenizer 编成 6 个 token:

```text
[A, B, C, D, E, F]
```

日志类似:

```text
input_len=6
prefix_len=0
input_ids.shape=[6]
extend_lens=[6]
prefill_rids=[rid]
mode='extend'
```

解释:

```text
prefix_len=0      没有命中 prefix cache
extend_lens=[6]   需要处理 6 个新 token
mode='extend'     Scheduler 认为这是 extend 阶段
prefill_rids      MLX runner 发现这是新请求,内部走 prefill_start
```

可以画成:

```text
KV cache 初始为空

prompt: A B C D E F
        └───────┬───────┘
              prefill

写入 KV slot: 1 2 3 4 5 6
产出第一个生成 token: G1
```

所以这一次在调度层叫:

```text
extend forward
```

在模型执行层叫:

```text
prefill
```

---

## 例子二:命中 prefix cache 后的 extend

第二个请求 prompt 是 5 个 token:

```text
[A, B, D, E, F]
```

假设前两个 token `[A, B]` 已经在 prefix cache 里。

日志类似:

```text
input_len=5
prefix_len=2
input_ids.shape=[3]
input_ids_head=[D, E, F]
extend_lens=[3]
#new-token: 3
#cached-token: 2
```

解释:

```text
input_len=5       原始 prompt 总长 5
prefix_len=2      前 2 个 token 命中 cache
input_ids=[D,E,F] 只需要新算后 3 个 token
extend_lens=[3]   本轮追加 3 个 token
```

图示:

```text
prompt: A B D E F
        │ │ └─┬─┘
        │ │   └── 本轮 extend/prefill 的新 token
        └─┴────── 已命中 prefix cache

cached-token: 2
new-token:    3
```

这里仍然是 `extend`,但不是从头 prefill 5 个 token,而是复用前 2 个 cache,只处理后 3 个。

---

## 例子三:Chunked Prefill

假设 prompt 很长,有 10000 个 token:

```text
T1 ... T10000
```

如果一次性 prefill 太大,调度器可能拆块:

```text
第 1 块: T1    ... T4096
第 2 块: T4097 ... T8192
第 3 块: T8193 ... T10000
```

每一块在 Scheduler 看来都是一次 `extend`:

```text
round 1: extend_lens=[4096]
round 2: extend_lens=[4096]
round 3: extend_lens=[1808]
```

但语义上,这些 `extend` 都是在完成同一个长 prompt 的 prefill,所以叫:

```text
chunked prefill
```

在 MLX worker 里大致对应:

```python
if self._mlx_runner.has_request(req.rid):
    if seq_len > 1:
        # 后续 chunk: chunked prefill continuation
        extend_start(...)
else:
    # 第一个 chunk: 新请求 prefill
    prefill_start(...)
```

图示:

```text
长 prompt:
T1 ........................................ T10000

拆块:
[chunk 1] [chunk 2] [chunk 3]
    │         │         │
    ▼         ▼         ▼
 extend    extend    extend
    │         │         │
    └─────────┴─────────┴── 都属于 chunked prefill
```

---

## Decode 是另一阶段

prefill / extend 完成后,模型已经有了 prompt 的 KV cache,并产出了第一个生成 token。

之后进入 decode:

```text
已有序列: A B C D E F G1

decode round 1: 输入 G1 -> 输出 G2
decode round 2: 输入 G2 -> 输出 G3
decode round 3: 输入 G3 -> 输出 G4
...
```

日志里类似:

```text
forward_mode='2'
mode='decode'
input_ids.shape=[1]
seq_lens=[7]
lazy_tokens_shape=(1,)
```

解释:

```text
forward_mode='2'    decode 阶段
input_ids.shape=[1] 本轮只喂 1 个 token
seq_lens=[7]        当前序列长度已经是 prompt + generated
```

decode 的特点:

| 项 | prefill / extend | decode |
|----|------------------|--------|
| 主要处理 | prompt token / 新追加 token 段 | 上一步生成的 token |
| 每轮 token 数 | 通常多个 | 通常 1 个 |
| 主要目的 | 建 KV cache,产出第一个 token | 持续生成后续 token |
| 成本形态 | 一次较重 | 多轮轻量但次数多 |

---

## 放回 MLX overlap 日志里看

你看到的模式大概是:

```text
1. tokenizer.send_one_request
2. mlx.async_forward.begin forward_mode='1'
3. mlx.async_extend.begin
4. mlx.async_extend.end prefill_rids=[...]
5. mlx.overlap.finalize.end next_token_ids=[G1]
6. Prefill batch ...
7. mlx.async_forward.begin forward_mode='2'
8. mlx.async_forward.decode
9. mlx.async_chained_decode
10. mlx.overlap.promote_chained
```

含义:

```text
1-6:  prefill / extend 阶段,处理 prompt,产出第一个 token
7+:   decode 阶段,每轮生成一个 token
9-10: MLX overlap 提前排队下一轮 decode,减少 GPU 空档
```

---

## 为什么 forward_mode 叫 extend,但日志里有 prefill_rids

因为这是两个视角:

```text
Scheduler: 我现在要给这个请求追加一段 token → extend
MLX runner: 这个 req_id 我还没见过,要创建 cache → prefill_start
```

所以同一轮可能同时出现:

```text
forward_mode='1'       # 调度层: extend
mode='extend'          # MLX async 接口: extend
prefill_rids=[rid]     # 具体请求: 新请求,内部走 prefill
```

这不是矛盾,只是层级不同。

---

## 一句话回顾

> `extend` 是 Scheduler 的通用动作:给请求序列追加一段 token。新请求的第一次 extend 内部就是 `prefill`;长 prompt 被拆成多次 extend 就是 `chunked prefill`;prompt 处理完以后,每轮只喂一个生成 token 的阶段叫 `decode`。
