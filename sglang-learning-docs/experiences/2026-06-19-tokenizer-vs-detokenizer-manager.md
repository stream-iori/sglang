# TokenizerManager 与 DetokenizerManager

> 源码基准：learning `2aa70e3eb4`，已合入 origin/main `6fc8d9da32`（2026-10-10）。

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

UTF-8、BPE 和完整字节示例见 [tokenizer 详解](../05-reference/tokenizer-internals.md)。Rust processor/renderer 是另一个服务处理路径，本地 Python 主线不因此消失。

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/managers/tokenizer_manager.py](../../python/sglang/srt/managers/tokenizer_manager.py) | 输入与响应协调 |
| [python/sglang/srt/managers/detokenizer_manager.py](../../python/sglang/srt/managers/detokenizer_manager.py) | offset 与增量解码 |
