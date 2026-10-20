# Prefill：算哪些 token，给它们哪些 KV 地址

> 源码基准：上游 `16eb87c06f`；本地集成 `4f0e1cbea2`（2026-10-20）。本地日志增强与验证范围见总索引和 Mac 验证记录。

Prefill 计算输入上下文；EXTEND 表示本轮向已有前缀追加输入位置。命中前缀时，只需算剩余部分。

```text
输入位置       0  1  2  3 | 4  5  6
token ID       a  b  c  d | e  f  g
命中 KV        ✓  ✓  ✓  ✓ | 本轮计算
物理槽位      91 23 44 18 | 70  8 62   （示意，可不连续）
                         ↑ out_cache_loc 描述本轮写位置
```

## 四层数据不能混成一个编号

| 数据 | 示例 | 含义 |
|---|---|---|
| input_ids | 151643、42 | 词表 ID，决定 embedding |
| positions | 0、1、2 | 序列位置，参与位置编码 |
| req_to_token | 请求行 + 位置 → 槽位 | 每个请求读历史 KV 的映射 |
| KVLocPlan / pool | 地址空间与物理索引 | 当前 runner/backend 怎么读写真正的存储 |

普通 Full attention 可以先用直接映射理解；统一/SWA/分页等路径还会涉及翻译，不应把所有 `out_cache_loc` 都当成通用物理地址。

## Admission 不只是“还有多少空槽”

```text
prefix match
  → 设备命中？主机命中需恢复？
  → PrefillBudget：本轮 token、总容量、分页取整、SWA/共享预算
  → 分配请求行与 KV slots
  → prepare_for_extend
  → ForwardBatch → Attention → KV 写入
```

| 问题 | 为什么必须检查 |
|---|---|
| token budget | 本轮不能无限塞长 prompt |
| page size | 需要的物理资源可能向上取整 |
| decode headroom | 已运行请求还要继续增长 |
| 可驱逐容量 | 某些缓存可回收，受保护的不能回收 |
| host restore | 主机命中仍需传回可计算的设备池 |
| SWA/SSM 资源 | 不能只检查 Full KV 计数 |

## Chunked prefill

假设没有命中，prompt 长 10，chunk 上限 4：

| 轮次 | 本轮输入位置 | 已有前缀长度 | 结果 |
|---|---|---|---|
| 1 | 0..3 | 0 | 保存中间状态，prompt 未完成 |
| 2 | 4..7 | 4 | 保存中间状态 |
| 3 | 8..9 | 8 | prompt 完成，可进入普通 decode |

这是输入 token 分块，不是把 Transformer 的层切成三份。`SPLIT_PREFILL` 的层分割路径是另一种机制。

## 两个请求合并时怎么算长度

假设 A 的输入长 6、有效命中 4；B 的输入长 3、无命中：

| 请求 | 输入总长 | 已有有效 KV | 本轮新增 query | packed 范围 |
|---|---|---|---|---|
| A | 6 | 4 | 2 | [0,2) |
| B | 3 | 0 | 3 | [2,5) |

```text
input_ids = [A4 A5 B0 B1 B2]
positions = [ 4  5  0  1  2]
extend lengths = [2, 3]     seq lengths = [6, 3]
```

模型只输入本轮 query；Attention 通过每个请求的映射读取有效历史 KV。B 的 position 从 0 起，不会因为 packed 偏移为 2 就改成 2。

## 预算与真实地址的分界

预算只回答“能不能接纳”，分配才产生本轮槽位。二者之间还可能有驱逐、页对齐、Host 恢复和特定模型的 replay 开销。

| 预算值 | 不能直接等同于 |
|---|---|
| 剩余 token budget | 设备实际可分配的所有子池容量 |
| 用户 max_new_tokens | 本轮必须一次性写入的 token 数 |
| 匹配 token 长度 | 所有组件已经可供当前 forward 读取 |
| Full 池空闲 | SWA/SSM 子池也足够 |

当前模型专属 encoder replay 行可能参与预算，不能只从 prompt 字符数估算占用。优先追 `PrefillBudget`、prefill admission 调用和 allocator 的共同约束。

## 练习

把上表的 A 命中改成 2，写出 packed 输入、position 和总 query 数。答案：A 新增 4，B 新增 3，总计 7；A 的 position 为 2..5，B 为 0..2。物理槽位仍需实际分配，不能由这些 position 数值推导。

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/managers/schedule_batch.py](../../python/sglang/srt/managers/schedule_batch.py) | prepare_for_extend 与 chunk 状态 |
| [python/sglang/srt/mem_cache/allocation.py](../../python/sglang/srt/mem_cache/allocation.py) | alloc_for_extend / assign_req_to_token_pool |
| [python/sglang/srt/mem_cache/prefill_budget.py](../../python/sglang/srt/mem_cache/prefill_budget.py) | Full/SWA/共享预算 |
| [python/sglang/srt/mem_cache/kv_loc_plan.py](../../python/sglang/srt/mem_cache/kv_loc_plan.py) | 地址空间绑定和读写 |
