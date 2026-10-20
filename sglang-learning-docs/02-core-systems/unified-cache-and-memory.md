# 统一缓存与内存：一棵树，多种状态

> 源码基准：上游 `16eb87c06f`；本地集成 `4f0e1cbea2`（2026-10-20）。本地日志增强与验证范围见总索引和 Mac 验证记录。

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
| Host / Storage | pool_host/、memory_pool_host.py、storage/ | Host 镜像、传输与持久化 |
| 构造 | kv_cache_configurator.py / kv_cache_builder.py | 模型和配置 → pools/cache |

上游 mem_cache README 同时展示分层方向。当前 `allocator/`、`pool_host/` 已存在，`pool/` 尚未建立；设备池仍从实际的 memory_pool 等文件进入。

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

## 一次缓存复用的完整生命周期

```text
① 请求输入 → 匹配 token 路径
② validators 检查 Full / SWA / Mamba 的有效状态
③ 锁定可复用状态 → 为未命中后缀分配空间
④ forward 读取前缀，写入后缀
⑤ 请求完成 → 按策略发布可复用状态、释放请求引用
⑥ 容量紧张 → 选可驱逐状态 → 执行释放/备份等动作
```

匹配决定“哪个前缀可用”，allocator 决定“后缀写在哪里”，pool 才持有真正字节。`cached_tokens` 是请求报告的复用计数，不能据此推导树节点数或所有子池占用。

## Host 池构造已拆开设备事实与策略

| 对象 | 保存什么 | 不负责什么 |
|---|---|---|
| DevicePoolInfo | 池标签、层关联、buffer info 等设备描述 | 选择 HTTP 请求 |
| MHABufferInfo | K/V 张量、布局、head geometry、dtype、可选 scales | 决定 Host 总容量 |
| MLABufferInfo | MLA 持久 buffers、层索引、原生 split 信息 | 把所有设备布局强制变成 MHA |
| MambaStateBufferInfo | checkpoint 的 temporal、conv 和附属状态 | 保存所有临时投机 workspace |
| assembler | 容量、层路由、draft 描述、传输输入组合 | 重新实现模型 forward |
| pool_host | Host 镜像及相关数据传输 | 自行推断整个模型的调度策略 |

```text
设备 pool.get_device_pool_infos()
             ↓ 持久张量与布局事实
hybrid_pool_assembler / linker_pool_assembler
             ↓ buffer_info + 容量 + 层范围等显式参数
pool_host/mha.py / mla.py / mamba.py / unified.py ...
             ↓
controller / storage backend
```

设备池可能重定位底层 backing，因此某些描述提供回调在传输时取得当前 buffers。缓存过期指针再复制会出错；不能把 buffer info 理解成永远不变的物理地址快照。

## 混合状态和 draft 的物理组织

`UnifiedKVPool` 与 `unified_draft_pool.py`、`layout/fused_draft.py` 处理支持配置下的物理布局。当前上游新增不对称 KV 行、draft 与 SWA/循环状态组合等能力。是否启用由模型、配置和 builder 决定，普通 Qwen3 MPS 没有覆盖这些组合。

| 需要区分 | 原因 |
|---|---|
| 模型的 KV head 数与每 allocator token 的物理行数 | 融合布局可能让两者不同 |
| model layer ID 与 pool-local layer index | 多池路由和传输各用自己的索引范围 |
| 虚拟 token ID 与物理槽位 | 压缩、翻译与重定位后不能直接混用 |
| target 与 draft | 可能共享布局资源，但读写视角、有效长度和生命周期不同 |

## 算一笔账

对普通 BF16 MHA，纯 KV 字节约为 `2 × 层数 × KV heads × head_dim × token 数 × 2 bytes`。前面的 2 表示 K 与 V。Qwen3-0.6B 用本地 config 算得 112 KiB/token，4096 逻辑 token 为 448 MiB；这不包含权重、workspace、索引和 allocator 额外开销。

混合状态、量化 scales、分页、物理行融合、draft 副本应分别计算，不能把这个公式直接套给 Mamba 或 MLA。

## 练习与答案

| 观察 | 应如何解释 |
|---|---|
| 同一前缀，Full 命中但 SWA 缺失 | 结构和部分组件匹配，不代表整个请求可直接复用 |
| 请求结束后 available 没完全增加 | 先查可驱逐缓存和锁，不立即判断泄漏 |
| Host 命中但 forward 仍等待 | 状态还需恢复并满足消费依赖 |
| 两个 runner 的索引值相同 | 还要比较 IdSpace 和翻译器，数值相同不代表同一地址 |

进一步读：[设备描述](../../python/sglang/srt/mem_cache/device_pool_info.py)、[Host 池目录](../../python/sglang/srt/mem_cache/pool_host)、[混合装配](../../python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py)、[draft 布局](../../python/sglang/srt/mem_cache/layout/fused_draft.py)。

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/mem_cache/registry.py](../../python/sglang/srt/mem_cache/registry.py) | cache 构造与组件选择 |
| [python/sglang/srt/mem_cache/unified_radix_cache.py](../../python/sglang/srt/mem_cache/unified_radix_cache.py) | I/O controller |
| [python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py](../../python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py) | Python TreeCore |
| [python/sglang/srt/mem_cache/unified_cache/components/README.md](../../python/sglang/srt/mem_cache/unified_cache/components/README.md) | 组件契约 |
| [python/sglang/srt/mem_cache/unified_memory_pool.py](../../python/sglang/srt/mem_cache/unified_memory_pool.py) | 物理统一池 |
| [python/sglang/srt/mem_cache/kv_loc_plan.py](../../python/sglang/srt/mem_cache/kv_loc_plan.py) | IdSpace、bind、read_table、write_ids |
