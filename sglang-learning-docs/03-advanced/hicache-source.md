# HiCache：命中主机状态还要恢复到设备

> 源码基准：learning `2aa70e3eb4`，已合入 origin/main `6fc8d9da32`（2026-10-10）。

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
| memory_pool_host.py | Host tensor、索引和布局 |
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

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/mem_cache/unified_radix_cache.py](../../python/sglang/srt/mem_cache/unified_radix_cache.py) | host 命中与 I/O 控制 |
| [python/sglang/srt/mem_cache/hybrid_cache/hybrid_cache_controller.py](../../python/sglang/srt/mem_cache/hybrid_cache/hybrid_cache_controller.py) | 多池传输 |
| [python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py](../../python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py) | pool 装配 |
| [python/sglang/srt/mem_cache/memory_pool_host.py](../../python/sglang/srt/mem_cache/memory_pool_host.py) | Host 内存 |
| [python/sglang/srt/mem_cache/storage](../../python/sglang/srt/mem_cache/storage) | Storage 实现 |
