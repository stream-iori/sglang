# Prefill：算哪些 token，给它们哪些 KV 地址

> 源码基准：learning `2aa70e3eb4`，已合入 origin/main `6fc8d9da32`（2026-10-10）。

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

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/managers/schedule_batch.py](../../python/sglang/srt/managers/schedule_batch.py) | prepare_for_extend 与 chunk 状态 |
| [python/sglang/srt/mem_cache/allocation.py](../../python/sglang/srt/mem_cache/allocation.py) | alloc_for_extend / assign_req_to_token_pool |
| [python/sglang/srt/mem_cache/prefill_budget.py](../../python/sglang/srt/mem_cache/prefill_budget.py) | Full/SWA/共享预算 |
| [python/sglang/srt/mem_cache/kv_loc_plan.py](../../python/sglang/srt/mem_cache/kv_loc_plan.py) | 地址空间绑定和读写 |
