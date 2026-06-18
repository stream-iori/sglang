# Week 2 详细讲义：Scheduler 和 RadixCache

> 目标：把“请求怎么排队、怎么成 batch、怎么复用 KV Cache”讲清楚。只会基础 Python 也能按步骤读。

## 一句话结论

| 问题 | 答案 |
|---|---|
| Scheduler 是什么？ | 一个无限循环：收请求、入队、选 batch、执行、处理结果。 |
| `waiting_queue` 是什么？ | 新请求排队区，还没完成 prefill。 |
| `running_batch` 是什么？ | 已经 prefill 完，正在 decode 的请求集合。 |
| RadixCache 是什么？ | 用 token 前缀树复用历史 KV Cache，避免重复算 prompt。 |
| 内存池管什么？ | 管 KV Cache 的物理位置，不管“文本内容”。 |

## 1. 本周只看这条主线

```mermaid
flowchart TD
    A["TokenizerManager"] -->|"TokenizedGenerateReqInput"| B["Scheduler.process_input_requests()"]
    B --> C["Scheduler.handle_generate_request()"]
    C --> D["Req"]
    D --> E["waiting_queue"]
    E --> F["Scheduler.get_next_batch_to_run()"]
    F -->|"新请求 prefill"| G["ScheduleBatch(EXTEND)"]
    F -->|"继续运行请求"| H["ScheduleBatch(DECODE)"]
    G --> I["Scheduler.run_batch()"]
    H --> I
    I --> J["Scheduler.process_batch_result()"]
    J -->|"未完成"| F
    J -->|"完成"| K["DetokenizerManager"]

    style F fill:#ff6b6b,color:#fff
    style I fill:#7bed9f,color:#000
```

## 2. Day-by-Day

| 天 | 目标 | 只读这些 | 验收 |
|---|---|---|---|
| Day 1 | 看懂 Scheduler 心跳 | `scheduler.py:event_loop_normal` | 能默写 `recv -> process -> schedule -> run -> result` |
| Day 2 | 看懂请求入队 | `process_input_requests`, `handle_generate_request`, `Req` | 能说清 `TokenizedGenerateReqInput -> Req -> waiting_queue` |
| Day 3 | 看懂 batch 选择 | `get_next_batch_to_run`, `ScheduleBatch` | 能区分 EXTEND/DECODE |
| Day 4 | 看懂 RadixCache | `radix_cache.py:match_prefix`, `insert`, `evict` | 能手画一棵 token 前缀树 |
| Day 5 | 看懂内存池关系 | `memory_pool.py`, `schedule_batch.py` 字段 | 能区分“逻辑请求”和“KV 物理槽位” |

## 3. 最小数据结构

| 名字 | 文件 | 你先理解成 |
|---|---|---|
| `TokenizedGenerateReqInput` | `managers/io_struct.py` | Tokenizer 发来的“已转 token 请求” |
| `Req` | `managers/schedule_batch.py` | Scheduler 内部的单个请求状态 |
| `ScheduleBatch` | `managers/schedule_batch.py` | 一批要执行的 `Req` |
| `ForwardMode` | `model_executor/forward_batch_info.py` | 本轮是 prefill 还是 decode |
| `RadixCache` | `mem_cache/radix_cache.py` | token 前缀 -> KV Cache 位置 |
| `TokenToKVPool` | `mem_cache/memory_pool.py` | token index -> 真实 KV 存储 |
| `ReqToTokenPool` | `mem_cache/memory_pool.py` | req index + token position -> token index |

## 4. Scheduler 心跳

源码：`python/sglang/srt/managers/scheduler.py:event_loop_normal()`

```text
while True:
  1. recv_requests()
  2. process_input_requests()
  3. get_next_batch_to_run()
  4. run_batch()
  5. process_batch_result()
```

| 步骤 | 大白话 | 结果 |
|---|---|---|
| `recv_requests` | 从 ZMQ 收包 | 得到外部消息 |
| `process_input_requests` | 按消息类型分发 | 新请求变 `Req` |
| `get_next_batch_to_run` | 决定下一轮算谁 | 得到 `ScheduleBatch` |
| `run_batch` | 调模型执行 | 得到 next token/logprob |
| `process_batch_result` | 更新请求状态 | 完成就输出，没完成继续 decode |

## 5. 请求状态变化

```mermaid
stateDiagram-v2
    [*] --> waiting_queue: 新请求
    waiting_queue --> running_batch: EXTEND / prefill 完成
    running_batch --> running_batch: DECODE 生成 1 token
    running_batch --> finished: stop / max_new_tokens / abort
    finished --> DetokenizerManager: BatchTokenIDOutput
```

| 状态 | 表示 |
|---|---|
| waiting | prompt 还没完整处理 |
| running | prompt 已处理，正在逐 token 生成 |
| finished | 命中 stop、长度上限、abort 等结束条件 |

## 6. EXTEND 和 DECODE

| 对比 | EXTEND / Prefill | DECODE |
|---|---|---|
| 谁触发 | 新请求 | 已运行请求 |
| 输入 token 数 | prompt 多个 token | 通常 1 个 token |
| KV Cache | 大量写入 | 读历史 + 追加新 token |
| 延迟影响 | 影响 TTFT | 影响 ITL/TPS |
| 代码判断 | `forward_mode.is_extend()` | `forward_mode.is_decode()` |

```text
prompt = [10, 20, 30]

EXTEND:
  输入 [10, 20, 30]
  写 KV for 10/20/30
  采样得到 40

DECODE:
  输入 [40]
  读 KV for 10/20/30
  写 KV for 40
  采样得到 50
```

## 7. RadixCache 细讲：前缀树如何复用 KV Cache

先记一句话：**RadixCache 不是存文本，也不是存 logits；它存的是“某段 token 前缀已经算好的 KV Cache 位置”。**

### 7.1 为什么需要 RadixCache

多轮对话常常长这样：

```text
第 1 轮 prompt:
[system, history_1, user_1]

第 2 轮 prompt:
[system, history_1, assistant_1, user_2]

第 3 轮 prompt:
[system, history_1, assistant_1, user_2, assistant_2, user_3]
```

前面大量 token 是重复的。如果每次都重新 prefill，GPU 会反复算同一段上下文。

| 没有 RadixCache | 有 RadixCache |
|---|---|
| 每个 prompt 从头算 | 命中公共前缀，少算一段 |
| 长系统提示反复消耗 prefill | system prompt 的 KV 可复用 |
| 多轮对话 TTFT 更高 | 命中越多，TTFT 越低 |

### 7.2 Radix Tree 是什么

普通 Trie 一条边通常存 1 个 token；Radix Tree 会把连续 token 压缩成一段。

```text
普通 Trie:
root -> 1 -> 2 -> 3 -> 4
             └-> 5 -> 6

Radix Tree:
root -> [1, 2]
          ├-> [3, 4]
          └-> [5, 6]
```

好处：

| 设计 | 好处 |
|---|---|
| 边上存 token 片段 | 树更浅 |
| 公共前缀只存一次 | 多请求复用同一段 KV |
| 节点边界可 split | 部分命中时能精确切开 |

```mermaid
flowchart TD
    R["root"] --> P["[1, 2]<br/>共享前缀"]
    P --> A["[3, 4]<br/>请求 A 剩余"]
    P --> B["[5, 6]<br/>请求 B 剩余"]

    style P fill:#ffa502,color:#000
```

### 7.3 TreeNode 里到底存什么

真实源码在 `python/sglang/srt/mem_cache/radix_cache.py:TreeNode`。

| 字段 | 大白话 |
|---|---|
| `key` | 这条边代表的 token 片段，例如 `[1,2]` |
| `value` | 这段 token 对应的 KV cache token indices |
| `children` | 后续分支 |
| `parent` | 父节点 |
| `lock_ref` | 有多少运行中请求正在引用它，非 0 不能驱逐 |
| `last_access_time` | LRU/优先级淘汰时使用 |
| `hit_count` | 命中次数，统计/策略可用 |

关键区分：

```text
key   = token 内容，用来匹配前缀
value = KV 位置，用来复用已经算好的 K/V
```

### 7.4 `match_prefix`：找最长已缓存前缀

例子：树里已有两条缓存：

```text
[1, 2, 3, 4]
[1, 2, 5, 6]
```

新请求：

```text
[1, 2, 3, 9, 10]
```

匹配过程：

| 步骤 | 当前树边 | 输入剩余 | 结果 |
|---|---|---|---|
| 1 | `[1,2]` | `[1,2,3,9,10]` | 全匹配，继续 |
| 2 | `[3,4]` | `[3,9,10]` | 只匹配 `[3]` |
| 3 | 命中结束 | 剩余 `[9,10]` | 最长命中 `[1,2,3]` |

命中后：

```text
hit prefix = [1, 2, 3]
miss suffix = [9, 10]
```

SGLang 可以复用 `[1,2,3]` 的 KV，只需要对 `[9,10]` 做新的 prefill。

```mermaid
flowchart TD
    A["新请求 tokens<br/>[1,2,3,9,10]"]
    B["RadixCache.match_prefix"]
    C["命中 prefix<br/>[1,2,3]"]
    D["未命中 suffix<br/>[9,10]"]
    E["复用 prefix 的 KV indices"]
    F["只为 suffix 分配新 KV 并 forward"]

    A --> B
    B --> C --> E
    B --> D --> F
```

### 7.5 节点 split：为什么匹配到一半要切开

如果树里原来是：

```text
root -> [1,2] -> [3,4]
```

新请求只命中 `[3]`，就要把 `[3,4]` 拆成：

```text
root -> [1,2] -> [3] -> [4]
```

这样后续请求 `[1,2,3,8]` 也能直接命中到 `[1,2,3]`。

| 不 split | split 后 |
|---|---|
| 命中边界卡在 `[3,4]` 内部 | 命中边界变成独立节点 `[3]` |
| 下次还要重新处理部分匹配 | 下次匹配更直接 |
| 树结构不够精细 | 前缀复用粒度更准 |

### 7.6 `insert`：把新算出的 KV 写进树

继续上面的例子。新请求 `[1,2,3,9,10]` 已经命中 `[1,2,3]`，模型只新算了 `[9,10]` 的 KV。

插入后树变成：

```text
root
└── [1,2]
    ├── [3]
    │   ├── [4]
    │   └── [9,10]
    └── [5,6]
```

注意：插入的不是“输出文本”，而是：

```text
token 片段 -> 这段 token 的 KV cache indices
```

### 7.7 `lock_ref`：为什么有些缓存不能删

当某个请求正在 decode，它还要继续读自己的历史 KV。如果这时把它命中的 RadixCache 节点驱逐掉，请求就坏了。

所以节点有 `lock_ref`：

| `lock_ref` | 含义 | 能不能驱逐 |
|---|---|---|
| `0` | 没有运行中请求引用 | 可以 |
| `> 0` | 有请求正在用 | 不可以 |

示例：

```text
请求 A 命中节点 [1,2,3]
  -> inc_lock_ref([1,2,3])
  -> [1,2,3] 和祖先节点 lock_ref +1

请求 A 完成
  -> dec_lock_ref([1,2,3])
  -> 引用计数 -1
```

### 7.8 `evict`：内存不够时删谁

KV Cache 在 GPU 显存里，空间有限。内存不够时，RadixCache 会找可驱逐叶子节点。

```mermaid
flowchart TD
    A["需要释放 N 个 token 的 KV 空间"]
    B["收集 evictable leaves"]
    C{"lock_ref == 0?"}
    D["按 eviction strategy 排序<br/>LRU/priority 等"]
    E["free node.value<br/>释放 TokenToKVPool 里的 KV"]
    F["删除叶子节点"]
    G{"父节点没有 child<br/>且 lock_ref == 0?"}
    H["父节点也成为候选"]

    A --> B --> C
    C -->|"否"| B
    C -->|"是"| D --> E --> F --> G
    G -->|"是"| H --> D
    G -->|"否"| A
```

删除时真正释放的是：

```text
node.value 指向的 KV token indices
```

也就是通知 `TokenToKVPoolAllocator.free(...)`：这些 KV 槽位可以复用了。

### 7.9 和 Scheduler 的关系

RadixCache 不主动跑模型，它只给 Scheduler 提供两个信息：

| 信息 | Scheduler 怎么用 |
|---|---|
| 命中了多少 prefix | 已命中的 token 不再重复 prefill |
| 命中的 KV indices 在哪里 | 填到请求的 KV 映射里，让 attention 能读历史 |

简化流程：

```text
新请求 tokens
  -> match_prefix(tokens)
  -> 得到 cached prefix KV indices
  -> 只为 miss suffix 分配 KV
  -> EXTEND 只算 miss suffix
  -> 请求完成或推进后 insert 新 KV
```

### 7.10 和 `TokenToKVPool` 的关系

| 组件 | 关心的问题 | 类比 |
|---|---|---|
| RadixCache | “哪些 token 前缀已经算过？” | 图书目录 |
| TokenToKVPool | “KV 数据具体放在哪个显存槽？” | 书架 |
| ReqToTokenPool | “某请求第几个 token 对应哪个槽？” | 借书记录 |

一条缓存记录可以理解成：

```text
RadixCache:
  key   = [1,2,3]
  value = [kv_slot_10, kv_slot_11, kv_slot_12]

TokenToKVPool:
  kv_slot_10 -> layer0/layer1/... 的 K/V tensor 位置
```

### 7.11 手算例子

按顺序插入三个请求：

```text
A = [1,2,3,4]
B = [1,2,5,6]
C = [1,2,3,9]
```

| 请求 | 命中 | 新算 | 插入后变化 |
|---|---|---|---|
| A | `[]` | `[1,2,3,4]` | 建出 `[1,2,3,4]` |
| B | `[1,2]` | `[5,6]` | 分叉出 `[5,6]` |
| C | `[1,2,3]` | `[9]` | `[3,4]` split 成 `[3] -> [4]`，再加 `[9]` |

最终树：

```text
root
└── [1,2]
    ├── [3]
    │   ├── [4]
    │   └── [9]
    └── [5,6]
```

| 操作 | 做什么 |
|---|---|
| `match_prefix` | 找输入 token 和树中已有 token 的最长公共前缀 |
| `insert` | 把新算出来的 KV 对应 token 插入树 |
| `evict` | 内存不够时，删掉可驱逐节点 |

## 8. RadixCache 和内存池别混

| 名字 | 管“什么” | 类比 |
|---|---|---|
| RadixCache | 哪些 token 前缀已经算过 | 书签目录 |
| ReqToTokenPool | 某个请求每个位置对应哪个 token slot | 座位表 |
| TokenToKVPool | 每个 token slot 的 KV 真放在哪 | 仓库货架 |

```mermaid
flowchart LR
    Req["Req"] -->|"req_pool_idx + token position"| R2T["ReqToTokenPool"]
    R2T -->|"token index"| T2K["TokenToKVPool"]
    T2K -->|"physical KV tensors"| GPU["GPU memory"]

    style R2T fill:#74b9ff,color:#000
    style T2K fill:#7bed9f,color:#000
```

## 9. 初学者读码顺序

不要从文件第 1 行读到最后一行。按下面顺序跳读：

| 顺序 | 命令 |
|---|---|
| 1. 找主循环 | `rg -n "def event_loop_normal" python/sglang/srt/managers/scheduler.py` |
| 2. 找请求处理 | `rg -n "def process_input_requests|def handle_generate_request" python/sglang/srt/managers/scheduler.py` |
| 3. 找 batch 选择 | `rg -n "def get_next_batch_to_run" python/sglang/srt/managers/scheduler.py` |
| 4. 找数据结构 | `rg -n "class Req|class ScheduleBatch" python/sglang/srt/managers/schedule_batch.py` |
| 5. 找 cache 三板斧 | `rg -n "def match_prefix|def insert|def evict" python/sglang/srt/mem_cache/radix_cache.py` |

## 10. 动手练习

| 练习 | 命令 | 看什么 |
|---|---|---|
| Scheduler demo | `python sglang-learning-docs/06_demo_scheduler.py` | waiting/running 如何变化 |
| Radix demo | `python sglang-learning-docs/06_demo_radix_cache.py` | prefix match 如何减少 miss suffix |
| 真实单测 | `PYTHONPATH="python" python/.venv/bin/python -m pytest test/registered/unit/mem_cache/test_radix_cache_unit.py -v` | SGLang RadixCache 的边界行为 |

## 11. 练习参考答案方向

| 问题 | 参考答案 |
|---|---|
| 为什么新请求先进入 `waiting_queue`？ | 因为 prompt 还没 prefill，不能直接 decode。 |
| 为什么 prefill 后进入 `running_batch`？ | 因为 KV Cache 已经有 prompt 的历史状态，后续只需逐 token decode。 |
| RadixCache 命中后省了什么？ | 省掉公共前缀 token 的 forward 计算和 KV 写入。 |
| 内存不足先做什么？ | 尝试驱逐 `lock_ref == 0` 的缓存节点，不能驱逐正在被请求引用的 KV。 |
| `lock_ref` 是什么？ | 节点被正在运行请求引用的次数，非 0 表示不能淘汰。 |

## 12. 本周验收

| 验收项 | 合格标准 |
|---|---|
| 主循环 | 能说清 5 步心跳 |
| 请求状态 | 能画出 waiting -> running -> finished |
| batch 类型 | 能解释 EXTEND/DECODE 的区别 |
| RadixCache | 能手画 `[1,2,3]`、`[1,2,4]` 的共享树 |
| 内存池 | 能区分 RadixCache、ReqToTokenPool、TokenToKVPool |
