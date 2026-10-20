# Mamba/SSM：缓存的是状态，不只是历史 K/V

> 源码基准：上游 `16eb87c06f`；本地集成 `4f0e1cbea2`（2026-10-20）。本地日志增强与验证范围见总索引和 Mac 验证记录。

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

## Checkpoint、活跃状态与临时 workspace

| 数据 | 生命周期 | 能否直接作为前缀状态保存 |
|---|---|---|
| 活跃递推状态 | 随当前请求推进 | 需保留对应 token 边界，不能任意原地共享 |
| 可复用 checkpoint | 某个可恢复边界的稳定状态 | 由组件的匹配/版本规则决定 |
| draft/verify 中间状态 | 候选执行期间 | 不能不经接受处理就当成正式状态 |
| convolution 等附属状态 | 与递推状态共同定义模型上下文 | 必须一起恢复，不能只复制 temporal |

`MambaStateBufferInfo` 显式描述 temporal、conv 与额外 checkpoint buffers。临时 speculative workspace 不属于 Host checkpoint；看见张量形状相似，不足以互相替代。

## 本轮延后 COW 的位置

当前 ModelRunner 在符合条件的 target extend 前，执行 deferred clear/COW。MambaPool 是物理存储，清零与复制前需要把逻辑 slot 翻译到物理 slot；int8 checkpoint 路径还可能先反量化到活跃状态。

```text
匹配到共享状态 → 记录 clear/COW 计划
    → forward stream 执行翻译/复制/清零
    → 对应模型层读取并推进状态
```

这里的时序能避免 scheduler 准备与 forward 读取发生竞争，不能把所有 copy 都提前移到任意 CPU 调度点。

## 练习

A、B 共享 checkpoint S，A 接受候选长度 2，B 接受长度 0。画出各自下一轮需要的状态版本。答案：二者不应被一个原地更新后的状态覆盖；未接受的候选不能推进已提交边界。

混合模型的内存实验应记录 Full token 容量、状态槽、checkpoint 和并发，分别观察哪个资源先耗尽。本地 Qwen3 不覆盖这些状态池。

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/mem_cache/unified_cache/components/mamba.py](../../python/sglang/srt/mem_cache/unified_cache/components/mamba.py) | checkpoint 与 COW |
| [python/sglang/srt/mem_cache/memory_pool.py](../../python/sglang/srt/mem_cache/memory_pool.py) | MambaPool / HybridReqToTokenPool |
| [python/sglang/srt/mem_cache/unified_memory_pool.py](../../python/sglang/srt/mem_cache/unified_memory_pool.py) | 统一状态池/allocator |
| [python/sglang/srt/mem_cache/kv_cache_builder.py](../../python/sglang/srt/mem_cache/kv_cache_builder.py) | 模型状态池构造 |
