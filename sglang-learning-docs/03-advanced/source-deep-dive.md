# 高级源码索引：每项优化改变了哪一层

> 源码基准：learning `2aa70e3eb4`，已合入 origin/main `6fc8d9da32`（2026-10-10）。

先读普通请求，再按被改变的状态进入专题。Mac MPS 跑通不代表下列高级机制已经验证。

| 专题 | 改变点 | 阅读入口 |
|---|---|---|
| [统一缓存](../02-core-systems/unified-cache-and-memory.md) | 前缀与状态的生命周期 | registry → cache → tree components |
| [SWA](swa-source.md) | KV 保留范围与地址空间 | SWAComponent → pool → KVLocPlan |
| [Mamba/SSM](mamba-source.md) | 递推状态和分叉 | MambaComponent → MambaPool |
| [HiCache](hicache-source.md) | 状态跨设备/Host/Storage 移动 | HybridCacheController / storage |
| [投机解码](speculative-source.md) | 一轮验证多个候选 | worker / spec_info / accept_lens |
| [PD](disaggregation-source.md) | 请求与 KV 跨实例衔接 | prealloc / transfer / conn |
| [多卡](multi-gpu.md) | 张量、层、请求或专家跨 rank | parallel_state / linear / runtime_context |
| [Rust 服务链](rust-serving-and-processing.md) | 编码、解析和协议边界 | server / processor / renderer |
| [Model Gateway](model-gateway.md) | 请求选择哪个 engine | sgl-model-gateway |

```text
请求控制 → Scheduler → ForwardBatch → 模型与采样
               │              │
          cache 生命周期    地址 / Runner / 并行
               │
      状态位置：设备 ↔ Host ↔ Storage ↔ 远端 worker
```

高级机制可以组合，但组合是否受支持要读参数解析和初始化检查。不要从各自单独可用推断任意组合可用。

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/mem_cache/registry.py](../../python/sglang/srt/mem_cache/registry.py) | 普通和特殊 cache factory |
| [python/sglang/srt/disaggregation/decode.py](../../python/sglang/srt/disaggregation/decode.py) | 跨实例请求准备 |
| [python/sglang/srt/speculative/spec_info.py](../../python/sglang/srt/speculative/spec_info.py) | 算法和模式 |
