# 统一缓存与内存：一棵树，多种状态

> 源码基准：learning `2aa70e3eb4`，已合入 origin/main `6fc8d9da32`（2026-10-10）。

统一缓存把“前缀树的逻辑”和“不同状态的资源管理”分开。一棵树管理 token 路径，Full/SWA/Mamba 组件分别维护自己的可用状态、锁和驱逐。

```text
Scheduler / cache registry
            │
            ▼
UnifiedRadixCache：调度 I/O、执行延后动作
            │
            ▼
TreeCore：匹配、分裂节点、锁、驱逐决策
            │
       ┌────┴─────────┬────────────┐
       ▼              ▼            ▼
 FullComponent   SWAComponent  MambaComponent
 全量 KV          窗口 KV       recurrent state
       │              │            │
       └───── allocator / pool ─────┘
                    │
               设备字节/主机副本
```

## 一棵树不意味着每个节点总有所有数据

```text
token 路径： [A B] → [C D] → [E]
节点 [C D]
  Full : device value / lock_ref
  SWA  : device value 或 tombstone / 自己的 lock_ref
  Mamba: 状态槽 / 自己的 lock_ref / copy-on-write
```

| 名词 | 大白话 | 读码时注意 |
|---|---|---|
| component_data | 节点上某类状态的数据 | Full/SWA/Mamba 分别查 |
| lock_ref | 正在使用，不能随便丢 | 组件锁和 host 锁有区别 |
| tombstone | 树节点还在，某组件的设备数据已没了 | 不能把结构命中当成状态可用 |
| CacheAction | 树决定后交给 controller 做的动作 | 包括释放、备份、重建等 |
| match validator | 检查此位置的状态能不能复用 | 所需组件都满足才推进有效命中边界 |

Full 的驱逐主要使用可驱逐叶集合；辅助组件维护各自 LRU。不能概括为“所有组件共用一个 LRU”。驱逐还涉及优先级和跨组件一致性。

## 存储分层与职责

| 层 | 当前实际文件 | 输入 → 输出 |
|---|---|---|
| 批次分配 | allocation.py | batch → 本轮写位置 |
| 空位管理 | allocator/ | 需要多少槽 → 索引 |
| 物理布局 | memory_pool.py / unified_memory_pool.py / swa_memory_pool.py 等 | layer + 索引 → tensor |
| 多池路由 | hybrid_cache/ | layer → 对应 pool |
| Host / Storage | memory_pool_host.py、storage/ | 恢复/备份状态字节 |
| 构造 | kv_cache_configurator.py / kv_cache_builder.py | 模型和配置 → pools/cache |

上游 mem_cache README 展示的是分层方向；当前有 `allocator/`，但还没有 `pool/` 和 `pool_host/` 目录。读码以实际文件为准。

## KVLocPlan：地址到底属于谁

```text
请求 token 位置
    → 请求映射 / 本轮虚拟写 ID
    → KVLocPlan 绑定 runner 和 IdSpace
    → Full / Sliding Window 的翻译
    → 对应池的读表与写索引
```

`IdSpaceKind` 中的 FULL 和 SLIDING_WINDOW 表达不同子池的读写视角。普通路径可能不需要额外翻译；复杂路径不可以只拿一组裸索引到处用。

## 和 UnifiedKVPool 的区别

| 名称 | 统一了什么 |
|---|---|
| UnifiedRadixCache | 前缀树、组件状态与生命周期 |
| UnifiedKVPool | 支持路径下的物理池组织与资源管理 |
| KVLocPlan | 某一轮、某个 runner 的地址表达 |

三者不是同一个对象；用了统一前缀树也不能据此推断所有物理池都启用统一内存方案。模型、配置、builder 决定具体池。

## 推荐观察

第一次同 prompt：设备命中低，执行较长 extend；第二次：命中增加，extend 缩短。请求结束后 pool 的“立即空闲”可能没有恢复到初始值，因为合法缓存还占着槽。

Rust TreeCore 可替换部分树操作，边界在 `tree_core_registry.py` 和 interface；本地默认类型必须由实际选择或日志确认。

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/mem_cache/registry.py](../../python/sglang/srt/mem_cache/registry.py) | cache 构造与组件选择 |
| [python/sglang/srt/mem_cache/unified_radix_cache.py](../../python/sglang/srt/mem_cache/unified_radix_cache.py) | I/O controller |
| [python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py](../../python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py) | Python TreeCore |
| [python/sglang/srt/mem_cache/unified_cache/components/README.md](../../python/sglang/srt/mem_cache/unified_cache/components/README.md) | 组件契约 |
| [python/sglang/srt/mem_cache/unified_memory_pool.py](../../python/sglang/srt/mem_cache/unified_memory_pool.py) | 物理统一池 |
| [python/sglang/srt/mem_cache/kv_loc_plan.py](../../python/sglang/srt/mem_cache/kv_loc_plan.py) | IdSpace、bind、read_table、write_ids |
