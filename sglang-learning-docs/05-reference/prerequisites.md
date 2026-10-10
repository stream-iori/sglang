# 前置知识：边读边补，不必先学完所有 CUDA

> 源码基准：learning `2aa70e3eb4`，已合入 origin/main `6fc8d9da32`（2026-10-10）。

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

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/managers/tokenizer_manager.py](../../python/sglang/srt/managers/tokenizer_manager.py) | 异步请求 |
| [python/sglang/srt/model_executor/forward_batch_info.py](../../python/sglang/srt/model_executor/forward_batch_info.py) | tensor 数据结构 |
