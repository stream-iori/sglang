# 投机、分布式与 PD：三个不同问题

> 源码基准：上游 `16eb87c06f`；本地集成 `4f0e1cbea2`（2026-10-20）。本地日志增强与验证范围见总索引和 Mac 验证记录。

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

## 组合时先检查增加了什么状态

| 组合 | 新增核对 |
|---|---|
| 投机 + TP | draft/target 通信组和各自执行时间 |
| 投机 + SWA/SSM | 候选状态、接受边界、窗口和 checkpoint |
| PD + HiCache | 远端接收与本地恢复的容量、锁和完成时序 |
| Rust 前端 + PD | 路由角色、typed 请求与对应调度/传输链 |

支持两个单项，不能推导它们组合受支持。先查参数解析拒绝条件、模型 override 和现有集成测试，再投入部署时间。

自测：给“吞吐下降”列一份最小证据表。答案应包含模型/提交/启动参数、输入输出长度、并发、各阶段耗时、队列、状态容量和错误日志；缺少这些，不能只按优化名称猜根因。

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/speculative/spec_info.py](../../python/sglang/srt/speculative/spec_info.py) | 当前算法列表 |
| [python/sglang/srt/distributed/parallel_state.py](../../python/sglang/srt/distributed/parallel_state.py) | 通信组 |
| [python/sglang/srt/disaggregation/prefill.py](../../python/sglang/srt/disaggregation/prefill.py) | prefill 侧 |
