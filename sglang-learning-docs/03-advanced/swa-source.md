# SWA：窗口内的 KV 和全量 KV 不同

> 源码基准：learning `2aa70e3eb4`，已合入 origin/main `6fc8d9da32`（2026-10-10）。

Sliding Window Attention 让某些层只看近期窗口。能释放哪些状态，取决于模型的 Attention 语义和缓存组件，而不是任意把老 KV 删除。

```text
序列位置：0 1 2 3 4 5 6 7 8 9
Full 层： [───────────────────] 读允许范围的全部历史
SWA 层：              [6 7 8 9] 假设有效窗口为 4（示意）
```

混合模型里 Full 和 SWA 层同时存在。一个 token 的 Full 数据和 SWA 数据可能在不同槽位和地址空间中。

| 组件 | 职责 |
|---|---|
| SWAComponent | 窗口有效性、锁、tombstone、组件驱逐 |
| SWAKVPool / UnifiedSWAKVPool | 实际状态布局与 Full/SWA 关系 |
| KVLocPlan 的 SLIDING_WINDOW | 给 runner/backend 提供正确的子池读写视角 |
| SWAPrefillBudget / SharedSWAPrefillBudget | admission 时判断各类或共享资源够不够 |

## 为什么树节点还在但不能全部复用

```text
相同 token 前缀 → 找到树节点
                  ├─ Full 数据在
                  └─ SWA 数据已驱逐，只有 tombstone
                          ↓
              仍要检查窗口条件 / 恢复 / 重算
```

Full 命中不等于该混合模型所需的所有状态都命中。统一树通过组件 validator 维护有效边界。

pure SWA 在当前 registry 有单独分支，不应把所有 SWA 配置都说成 UnifiedRadixCache。

本地 Qwen3-0.6B 的配置 `use_sliding_window=false`，所以本地普通生成不是 SWA 验证。

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/mem_cache/unified_cache/components/swa.py](../../python/sglang/srt/mem_cache/unified_cache/components/swa.py) | 组件生命周期 |
| [python/sglang/srt/mem_cache/swa_memory_pool.py](../../python/sglang/srt/mem_cache/swa_memory_pool.py) | SWA 物理池 |
| [python/sglang/srt/mem_cache/unified_memory_pool.py](../../python/sglang/srt/mem_cache/unified_memory_pool.py) | 统一 SWA 池 |
| [python/sglang/srt/mem_cache/kv_loc_plan.py](../../python/sglang/srt/mem_cache/kv_loc_plan.py) | SLIDING_WINDOW 地址视角 |
| [python/sglang/srt/mem_cache/registry.py](../../python/sglang/srt/mem_cache/registry.py) | pure SWA 特例 |
