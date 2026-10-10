# 调度与缓存：先学决策，再学复用

> 源码基准：learning `2aa70e3eb4`，已合入 origin/main `6fc8d9da32`（2026-10-10）。

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

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/mem_cache/registry.py](../../python/sglang/srt/mem_cache/registry.py) | 真实 factory 分支 |
| [python/sglang/srt/managers/schedule_policy.py](../../python/sglang/srt/managers/schedule_policy.py) | 排序和 prefill admission |
| [python/sglang/srt/mem_cache/prefill_budget.py](../../python/sglang/srt/mem_cache/prefill_budget.py) | 资源约束 |
| [python/sglang/srt/mem_cache/unified_radix_cache.py](../../python/sglang/srt/mem_cache/unified_radix_cache.py) | 前缀复用生命周期 |
