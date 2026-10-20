# SWA：窗口内的 KV 和全量 KV 不同

> 源码基准：上游 `16eb87c06f`；本地集成 `4f0e1cbea2`（2026-10-20）。本地日志增强与验证范围见总索引和 Mac 验证记录。

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

## 一次窗口推进的资源变化

以概念窗口 4 为例，位置 8 的 query 需要哪些历史位置由模型 mask 定义。不要忽略“窗口是否包含当前 token”的约定差异；图中的 4 仅表示一个允许范围示例。

```text
旧步有效窗口 → 新 query 到达 → 新槽写入 → 旧窗口部分不再需要
                                      ↓
                组件锁/投机读窗口/传输仍可能延长某些槽的寿命
```

Full 层仍可能需要全部历史，因此不能因为 SWA 层过窗，就释放同一 token 的所有组件。

## 当前组合路径

当前统一物理布局支持更多 draft、SWA 与 recurrent-state 组合。`KVLocPlan` 需要按 runner 和子池提供读写视角；投机 verify 的读取范围可能超过已经提交的 seq_len，应读 `window_read_extent` 等实际计算，避免机械截断。

| 检查点 | 应证明 |
|---|---|
| registry | 当前是 pure SWA 特例还是统一组件 |
| budget | Full/SWA 是独立资源还是共享预算 |
| translator | window indices 属于正确子池 |
| validator | 命中范围仍满足窗口与状态条件 |
| release | 过窗回收不会破坏锁定或仍在读取的数据 |

验收：解释同一 token 在 Full 与 SWA 子池有不同存活期的原因，再把这个结论对应到组件状态与地址，而不是只画一个环形数组。

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/mem_cache/unified_cache/components/swa.py](../../python/sglang/srt/mem_cache/unified_cache/components/swa.py) | 组件生命周期 |
| [python/sglang/srt/mem_cache/swa_memory_pool.py](../../python/sglang/srt/mem_cache/swa_memory_pool.py) | SWA 物理池 |
| [python/sglang/srt/mem_cache/unified_memory_pool.py](../../python/sglang/srt/mem_cache/unified_memory_pool.py) | 统一 SWA 池 |
| [python/sglang/srt/mem_cache/kv_loc_plan.py](../../python/sglang/srt/mem_cache/kv_loc_plan.py) | SLIDING_WINDOW 地址视角 |
| [python/sglang/srt/mem_cache/registry.py](../../python/sglang/srt/mem_cache/registry.py) | pure SWA 特例 |
