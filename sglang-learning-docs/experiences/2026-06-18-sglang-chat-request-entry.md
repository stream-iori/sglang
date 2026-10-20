# Chat 请求：messages 先变成模型输入

> 源码基准：上游 `16eb87c06f`；本地集成 `4f0e1cbea2`（2026-10-20）。本地日志增强与验证范围见总索引和 Mac 验证记录。

```text
POST /v1/chat/completions
  → HTTP route / OpenAI serving
  → 参数验证 + chat template + 编码
  → GenerateReqInput / TokenizerManager.generate_request
  → Scheduler → 模型 → 输出处理
  → ChatCompletion 或 SSE chunks
```

| 原始形态 | 下一步 |
|---|---|
| messages | 模型 template 加角色/控制符 |
| 渲染文本或编码结果 | tokenizer / token IDs |
| sampling 字段 | 转为引擎采样参数 |
| 输出 IDs | 增量解码、reasoning/tools、OpenAI 响应 |

某些路径已经带 input_ids，不能假定每层都重复分词。当前 chat_encoding 拆分了承载编码的职责；读 handler 时还要追这些辅助模块。

本地 Qwen3 验证用 `chat_template_kwargs.enable_thinking=false` 做简短算术检查；这不是模型所有聊天能力的评测。

## 一个请求有三层协议

| 层 | 输入例子 | 输出例子 |
|---|---|---|
| OpenAI API | messages、max_tokens、stream | ChatCompletion/SSE |
| 引擎请求 | token IDs、sampling、rid | 输出 ID 与元数据 |
| 模型计算 | packed IDs、positions、KV 地址 | logits 和采样结果 |

练习：`enable_thinking=false` 应先在哪层理解？答案：模型 chat template/编码路径，不是删除 Scheduler 的某种“思考线程”。改变模板可能改变输入 token 和缓存键，不能只比较用户正文。

流式结果需拼接 delta.content，并处理 DONE；某些请求含 reasoning/tools 时还要读对应协议字段。本地简短算术用例不覆盖全部工具调用能力。

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/entrypoints/http_server.py](../../python/sglang/srt/entrypoints/http_server.py) | HTTP route |
| [python/sglang/srt/entrypoints/openai/serving_chat.py](../../python/sglang/srt/entrypoints/openai/serving_chat.py) | chat serving |
| [python/sglang/srt/entrypoints/openai/chat_encoding.py](../../python/sglang/srt/entrypoints/openai/chat_encoding.py) | 编码辅助 |
| [python/sglang/srt/managers/tokenizer_manager.py](../../python/sglang/srt/managers/tokenizer_manager.py) | generate_request |
