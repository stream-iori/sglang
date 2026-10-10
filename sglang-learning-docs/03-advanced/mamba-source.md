# Mamba/SSM：缓存的是状态，不只是历史 K/V

> 源码基准：learning `2aa70e3eb4`，已合入 origin/main `6fc8d9da32`（2026-10-10）。

递推模型用当前输入更新一份历史摘要状态。混合模型可以同时包含 Attention KV 和递推状态，不能全部套用普通 MHA 的每 token KV 公式。

```text
Attention：历史位置 → K/V 序列 → 当前 query 读取
SSM：     state(t) + input(t+1) → state(t+1)
混合：    两种状态一起维护，前缀复用必须满足所需状态条件
```

| 当前源码对象 | 作用 |
|---|---|
| MambaComponent | 统一树里的 checkpoint、锁和分叉状态管理 |
| MambaPool | 存 recurrent / convolution 等状态，布局由模型定义 |
| HybridReqToTokenPool | 请求的 KV 映射和状态槽关联 |
| UnifiedMambaSlotAllocator | 支持路径下的状态槽资源管理 |
| checkpoint / track buffers | 保存可复用或投机过程需要的状态版本 |

## 为什么要 copy-on-write

```text
A 与 B 共享前缀 P 的状态 S
    A 下一步 x：不能原地把 S 改成 Sx，再让 B 读它
    B 下一步 y：需要仍从 S 出发

分叉时保留共享版本 → 给需要修改的一侧独立状态槽 → 更新
```

Attention 的历史 KV 通常按位置追加；递推状态会被更新。相同“共享前缀”在两者上的写入风险不同。

## 容量怎么判断

| 容量 | 常用度量 |
|---|---|
| Attention KV | token/page 槽和字节 |
| SSM state | 状态槽数量、每槽形状、dtype |
| checkpoint/投机版本 | 额外状态副本及存活时间 |

读 builder 和 allocator 的实际分配，不能只用“上下文长度 × KV 字节”推断混合模型的总内存。

本地 Qwen3-0.6B 是 Attention 模型，未实测 Mamba/SSM。

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/mem_cache/unified_cache/components/mamba.py](../../python/sglang/srt/mem_cache/unified_cache/components/mamba.py) | checkpoint 与 COW |
| [python/sglang/srt/mem_cache/memory_pool.py](../../python/sglang/srt/mem_cache/memory_pool.py) | MambaPool / HybridReqToTokenPool |
| [python/sglang/srt/mem_cache/unified_memory_pool.py](../../python/sglang/srt/mem_cache/unified_memory_pool.py) | 统一状态池/allocator |
| [python/sglang/srt/mem_cache/kv_cache_builder.py](../../python/sglang/srt/mem_cache/kv_cache_builder.py) | 模型状态池构造 |
