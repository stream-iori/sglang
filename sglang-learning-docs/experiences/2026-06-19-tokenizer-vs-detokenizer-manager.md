# TokenizerManager 与 DetokenizerManager

> 源码基准：上游 `16eb87c06f`；本地集成 `4f0e1cbea2`（2026-10-20）。本地日志增强与验证范围见总索引和 Mac 验证记录。

```text
文字/messages → TokenizerManager → token ID / 请求 → Scheduler
文字片段      ← DetokenizerManager ← 输出 ID / 偏移 ← Scheduler
```

| 对象 | 核心职责 |
|---|---|
| TokenizerManager | 编码、请求校验/转发、异步响应管理等 |
| DetokenizerManager | ID 到文字、增量输出与上下文/偏移 |
| OpenAI serving | template、协议、reasoning/tools 等处理边界 |

当前 detokenizer 中的 `surr_offset` 保留上下文起点，`read_offset` 表示已读边界。用上下文重新 decode，再只追加新确认文字，可以避免重复与不完整输出；不是逐个 decode(token) 后简单拼接。

UTF-8、BPE 和完整字节示例见 [tokenizer 详解](../05-reference/tokenizer-internals.md)。Rust server/processor 是另一个服务处理路径，独立 renderer 已移除，本地 Python 主线不因此消失。

## 哪些工作并不对称

输入可一次拿到完整 prompt；输出往往逐 token 到达。完整 prompt 编码和增量解码因此不只是把箭头反过来。

| 输入侧 | 输出侧 |
|---|---|
| messages → template → IDs | IDs → 带上下文解码 → 新文本 |
| 长度、输入格式检查 | 停止、增量去重与尾部处理 |
| 请求转发与异步等待 | 响应片段与 finish reason |

练习：一个汉字分成两个 token，第一段 decode 出替换字符，能立即发给客户端吗？答案：需按实际增量实现保留/重试可确认边界，不能逐 token decode 后简单相加。

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/managers/tokenizer_manager.py](../../python/sglang/srt/managers/tokenizer_manager.py) | 输入与响应协调 |
| [python/sglang/srt/managers/detokenizer_manager.py](../../python/sglang/srt/managers/detokenizer_manager.py) | offset 与增量解码 |
