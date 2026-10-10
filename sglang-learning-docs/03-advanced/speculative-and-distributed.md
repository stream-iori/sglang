# 投机、分布式与 PD：三个不同问题

> 源码基准：learning `2aa70e3eb4`，已合入 origin/main `6fc8d9da32`（2026-10-10）。

| 机制 | 要解决的问题 | 增加的代价 |
|---|---|---|
| Speculative decoding | 每次大模型调用能否确认更多 token | draft、verify、候选状态和回滚 |
| TP/EP/PP 等 | 单个模型如何跨设备执行 | 通信、同步和负载均衡 |
| DP | 更多请求由多个副本处理 | 路由和副本内存 |
| PD 分离 | prefill/decode 能否分别扩缩容 | KV 传输、预分配、排队和故障处理 |

```text
投机：draft 候选 → target verify → 接受有效前缀 + 后续 token
TP： 同一层分片 → 通信组合结果
PP： 不同层分到不同 rank → 传中间激活
PD： prefill 实例 → 传 KV / metadata → decode 实例
```

它们不自动提高所有负载的速度。例如短 prompt 的 PD 传输成本可能不划算；draft 接受率低时，投机额外工作可能抵消收益。

学习顺序：[投机源码](speculative-source.md) → [多卡](multi-gpu.md) → [PD 源码](disaggregation-source.md)。所有性能结论需要实际模型、硬件和负载测量。

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/speculative/spec_info.py](../../python/sglang/srt/speculative/spec_info.py) | 当前算法列表 |
| [python/sglang/srt/distributed/parallel_state.py](../../python/sglang/srt/distributed/parallel_state.py) | 通信组 |
| [python/sglang/srt/disaggregation/prefill.py](../../python/sglang/srt/disaggregation/prefill.py) | prefill 侧 |
