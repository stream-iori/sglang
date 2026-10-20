# 前置知识：边读边补，不必先学完所有 CUDA

> 源码基准：上游 `16eb87c06f`；本地集成 `4f0e1cbea2`（2026-10-20）。本地日志增强与验证范围见总索引和 Mac 验证记录。

| 需要会什么 | 最小程度 | 对应阅读 |
|---|---|---|
| Python | 引用、类、async、import、类型注解 | experiences/ |
| HTTP/SSE | 请求响应、逐段返回、结束标记 | server-and-benchmark |
| 多进程/IPC | 地址空间独立、消息不是共享对象 | ZMQ demo |
| tensor | shape、dtype、device、索引 | math-for-llm |
| Transformer | Q/K/V、logits、采样 | transformer |
| tokenizer | token 与字符/byte 不同 | tokenizer-internals |
| GPU 基础 | 队列、kernel、同步 | cuda-graph-and-padding |

```text
能跑一个请求
  → 能画出请求路径
  → 能指出每个张量的 shape/用途
  → 能说明缓存命中与释放
  → 再研究 Graph / 多卡 / PD
```

本地 Mac 使用 [Torch MPS 指南](../setup/mac-debug.md)。不要为了读普通请求链先安装 CUDA/Triton 的所有扩展；高级算子执行需要匹配平台。

Python 基础笔记的文件名保留原日期以兼容链接，内容已按本次源码学习路径整理；它们不是对应日期的旧运行结论。

## 可按八个学习单元推进

| 单元 | 学习内容 | 交付 |
|---|---|---|
| 1 | Python 最小语法、环境与启动 | 第一份服务验证目录 |
| 2 | HTTP、IPC、请求和输出 | 请求进程图 |
| 3 | Req、batch、normal/overlap | 双请求时序表 |
| 4 | Prefill/Decode、KV 地址 | packed 输入与映射图 |
| 5 | 统一缓存、Host/SSM 状态 | 生命周期与容量账本 |
| 6 | 模型、配置与执行器 | forward 分支与配置来源 |
| 7 | 一项高级机制 | 源码报告与硬件实验设计 |
| 8 | 复现、定位和小修改 | 结业报告及相关回归证据 |

每个单元可用多次学习完成，不规定一周必须读完多少行。若只能解释字段名却不能预测例子中的变化，回到实例和实验；不要用继续翻页替代理解。

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/managers/tokenizer_manager.py](../../python/sglang/srt/managers/tokenizer_manager.py) | 异步请求 |
| [python/sglang/srt/model_executor/forward_batch_info.py](../../python/sglang/srt/model_executor/forward_batch_info.py) | tensor 数据结构 |
