# HiCache：命中主机状态还要恢复到设备

> 源码基准：上游 `16eb87c06f`；本地集成 `4f0e1cbea2`（2026-10-20）。本地日志增强与验证范围见总索引和 Mac 验证记录。

HiCache 延长缓存的保存层级。数据在 Host/Storage 命中，可以省重算，但计算前仍要恢复设备可用状态。

```text
L1 设备 pool  ←──── load back ────  L2 Host pool  ←── prefetch ── L3 Storage
              ───── backup ─────→               ─── persist ──→
                          token/hash 与各级索引协调
```

## 当前阅读入口

| 层 | 看什么 |
|---|---|
| UnifiedRadixCache | host 命中、pending I/O、延后动作、锁 |
| HybridCacheController | 多池 Host/设备复制和消费时序 |
| hybrid_pool_assembler | 不同 layer/state 的池怎样组合 |
| device_pool_info.py / pool_host/ | 持久设备描述、Host tensor、索引与布局 |
| storage/ | 后端各自的存在性、读写、连接和生命周期 |

统一缓存下的 HiCache 要从当前 controller/组件链进入；存在旧 hiradix_cache 文件不代表它是本次运行使用的实例。

## 一次 host hit

```text
prefix match → host 有状态 → 锁定/预算检查 → 预留设备槽
    → H2D load back → 完成信号/消费依赖 → ForwardBatch 可以读
```

“命中”说明找到状态；“恢复完成”说明设备能用。把两者混同会造成读取尚未传完的数据。

| 现象 | 取证方向 |
|---|---|
| 命中高，TTFT 仍大 | load-back/prefetch 时间、带宽、排队 |
| Host 索引被复用后数据错 | host 锁、完成信号、索引释放时机 |
| 部分层可恢复，整体不可用 | 所需 Full/SWA/Mamba 组件完整性 |
| 存储后端启动失败 | 依赖、连接、路径、当前 backend 选择 |

Mac 当前验证没有启用 HiCache，也没有验证 Storage 后端。

## 新 Host 池边界为什么值得重读

设备 provider 描述自己的已有 buffers；assembler 决定 Host 容量与层路由，再把 `buffer_info` 等输入交给 Host 池。这样 Host 池不必依赖整个设备池的内部属性，但保留原生布局、strides、scales 和指针生命周期仍然重要。

```text
provider.get_device_pool_infos()
    → assembler：设备描述 + 容量策略 + draft/层路由
    → pool_host：镜像和传输
    → controller：提交、等待、消费、释放
```

当前 `pool_host/` 已存在；`memory_pool_host.py` 不能再作为全部 Host 实现的唯一入口。新 `hicache_key_scheme.py` 还涉及存储键，Mamba 等状态的 key 必须区分需要区分的 rank/状态，不能把不同来源混写。

## 传输顺序也是正确性

| 顺序 | 必须先发生 |
|---|---|
| 设备写回 Host | 生产这份 KV 的 forward 完成必要写入 |
| 从 Host 读回设备 | Host 状态有效、接收容量已预留 |
| 图重放读取 KV | 对应层数据已加载且消费依赖成立 |
| 复用传输槽位 | 使用它的传输和模型读取已经安全结束 |

只看带宽正常不能证明这些顺序正确。排查错误时同时记录事件、stream、slot 和请求关联。

## 实验设计

用支持 HiCache 的 GPU 环境分别测：冷输入、设备热缓存、设备驱逐后 Host 命中、Storage 恢复。每组保留输出 ID、命中层级和时间，而非只报总 hit rate。

验收：能解释“复用省重算，但恢复耗时可能增加 TTFT”，并给出能区分计算与传输的证据。本章没有在 Mac 上运行 Host/Storage 实验。

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/mem_cache/unified_radix_cache.py](../../python/sglang/srt/mem_cache/unified_radix_cache.py) | host 命中与 I/O 控制 |
| [python/sglang/srt/mem_cache/hybrid_cache/hybrid_cache_controller.py](../../python/sglang/srt/mem_cache/hybrid_cache/hybrid_cache_controller.py) | 多池传输 |
| [python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py](../../python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py) | pool 装配 |
| [python/sglang/srt/mem_cache/memory_pool_host.py](../../python/sglang/srt/mem_cache/memory_pool_host.py) | Host 内存 |
| [python/sglang/srt/mem_cache/storage](../../python/sglang/srt/mem_cache/storage) | Storage 实现 |
