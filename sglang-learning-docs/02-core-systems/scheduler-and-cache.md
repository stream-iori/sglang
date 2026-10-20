# 调度与缓存：先学决策，再学复用

> 源码基准：上游 `16eb87c06f`；本地集成 `4f0e1cbea2`（2026-10-20）。本地日志增强与验证范围见总索引和 Mac 验证记录。

Scheduler 决定“本轮让谁算”；prefix cache 决定“哪些状态已经算过”；allocator 决定“哪里有空位”。三个问题相互影响，但职责不同。

```text
等待请求 → 调度策略/预算 → 本轮 batch
               ↑              ↓
          prefix match      KV slots 分配
               ↑              ↓
       UnifiedRadixCache ← 完成/中间结果
               │
      TreeCore + Full/SWA/Mamba
```

## 当前默认缓存入口

`mem_cache/registry.py:default_radix_cache_factory` 的普通 Full attention 路径进入 `create_unified_radix_cache`。本地 Qwen3-0.6B 日志也确认 Unified Radix Cache。

| 条件 | 应追的分支 |
|---|---|
| 普通 Full / 支持的混合模型 | UnifiedRadixCache 及组件 |
| pure SWA | PureSWARadixCache 分支 |
| 指定 cache backend / 外部存储 | registry、配置和相应 factory |
| disable radix cache | 检查工厂和缓存行为；不等于完全不用 pool |

旧 `radix_cache.py` 仍可用于理解基础树和历史测试，但不能直接当作本地默认实例。

## 一次命中的作用

| 输入 | 缓存 | 新计算 |
|---|---|---|
| A B C D E | A B C | D E |
| A B X Y | A B C | X Y |
| A B C（再次请求） | A B C | 实际可复用边界还受 logits、分页和组件有效性约束 |

缓存以模型输入 token 前缀及相关 key 为依据，不是任意相似中文句子都能共享。请求结束后“留下缓存”和“释放请求行”可以同时发生。

## 学习顺序

| 次序 | 文档 | 要回答的问题 |
|---|---|---|
| 1 | [生命周期](scheduler-batch-lifecycle.md) | 请求什么时候进入/退出运行集合 |
| 2 | [Prefill](prefill-batch-and-kv.md) | 输入片段怎么拿到写地址 |
| 3 | [统一缓存](unified-cache-and-memory.md) | Full/SWA/Mamba 怎样保护和驱逐 |
| 4 | [状态日志](scheduler-status-log-scenarios.md) | 如何用证据确认命中与内存状态 |

## 把“先服务谁”和“复用什么”拆开

| 决策 | 输入 | 输出 |
|---|---|---|
| schedule policy | waiting queue、优先级、已有前缀信息 | 候选顺序 |
| admission / budget | 候选、运行请求、各类容量 | 本轮可接纳集合 |
| cache match | token 路径、所需组件、状态位置 | 有效命中边界及待恢复信息 |
| allocation | 需新增的位置、页和池约束 | 请求行、本轮写地址 |

缓存命中可能改变处理一个请求的成本，但不能把策略、命中和分配画成同一个函数。Full 空闲足够也不保证 SWA/SSM 预算足够。

## 场景推演

A 是短请求但冷缓存；B 是长请求但大部分前缀可复用。谁先运行取决于策略、优先级和预算，不能只按原始 prompt 长短判断。

验收：从日志中找一个请求，分别写下它的等待状态、有效命中、本轮 extend 长度与最终生成长度；四个数字描述不同的工作，不能互换。

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/mem_cache/registry.py](../../python/sglang/srt/mem_cache/registry.py) | 真实 factory 分支 |
| [python/sglang/srt/managers/schedule_policy.py](../../python/sglang/srt/managers/schedule_policy.py) | 排序和 prefill admission |
| [python/sglang/srt/mem_cache/prefill_budget.py](../../python/sglang/srt/mem_cache/prefill_budget.py) | 资源约束 |
| [python/sglang/srt/mem_cache/unified_radix_cache.py](../../python/sglang/srt/mem_cache/unified_radix_cache.py) | 前缀复用生命周期 |
