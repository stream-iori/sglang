# Profiling：先定位哪段慢，再解释原因

> 源码基准：learning `2aa70e3eb4`，已合入 origin/main `6fc8d9da32`（2026-10-10）。

慢请求先拆成排队、准备、模型、传输和输出，不要一开始就归咎于某个 kernel。

```text
HTTP 总延迟
  ├─ template / tokenize
  ├─ queue / admission / restore / transfer
  ├─ forward + sampling
  └─ detokenize / SSE / 网络
```

| 证据 | 回答的问题 | 局限 |
|---|---|---|
| 响应时间与 token 统计 | 用户感知是否变慢 | 不能指出 kernel 根因 |
| request trace | 慢在哪个阶段 | span 粒度/level 影响可见性 |
| scheduler.status | 当时 batch、queue、pool 的状态 | 快照采样，不是每次变更 |
| Torch/device profile | 哪些算子、同步、通信耗时 | 有观测开销；MPS/CUDA 能力不同 |
| CPU stack | Python 在忙什么或等什么 | 不等于设备执行时间 |

## 可靠对照

1. 固定模型、版本、backend、prompt/output 长度、sampling 和负载。
2. 预热；把冷启动、冷缓存和稳态分别记录。
3. 保存原始 profile 和请求响应。
4. 改一个变量，先检查正确性，再比较端到端和分阶段耗时。

测 CUDA kernel 用 CUDA 对应工具；本地 MPS profile 不会产生真实 CUDA kernel 时间。同步会改变异步时间线，不能用某段 Python wall time 直接当 kernel 时间。

Graph 的收益需同时记录提交时间、设备计算、padding、capture 开销和显存；长 GEMM 的相对收益小不代表 Graph 一定更慢。

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/managers/scheduler_components/profiler_manager.py](../../python/sglang/srt/managers/scheduler_components/profiler_manager.py) | profiling 控制 |
| [python/sglang/srt/observability/req_time_stats.py](../../python/sglang/srt/observability/req_time_stats.py) | 请求阶段 |
| [python/sglang/srt/observability/trace.py](../../python/sglang/srt/observability/trace.py) | trace 输出 |
| [python/sglang/srt/model_executor/runner](../../python/sglang/srt/model_executor/runner) | 执行器 |
