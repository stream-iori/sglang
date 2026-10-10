# 练习答案与验证边界

> 源码基准：learning `2aa70e3eb4`，已合入 origin/main `6fc8d9da32`（2026-10-10）。

| 练习 | 答案 | 为什么 |
|---|---|---|
| 请求 A 结束、B 未结束 | 下一轮保留 B，A 退出运行集合 | batch 不是固定队伍 |
| [1,2,0] 的最长已缓存前缀 | [1,2] | 从第三个位置分叉，不能复用旧第三个状态 |
| 玩具 scheduler 返回 batch 与 SRT 比较 | SRT 当前返回 plan，再取 batch_to_run | API 不能照旧例子假设 |
| ord('你') 能表示真实 token 吗 | 不能，它是 Unicode 码点 | tokenizer 是词表/规则，不是 ord |
| cold/warm 输出 IDs 相同 | 需要 deterministic 配置和相同输入 | 概率采样可能不同 |
| warm cached_tokens 增长 | 是前缀复用证据之一 | 仍需看模型/配置与响应 |

## 本地参考结果

在已保存的 Qwen3-0.6B MPS 验证中：198-token prompt 首次 cached=0，第二次 cached=197，两次 8-token 输出 IDs 相同；chat/SSE 输出 `2 + 2 = 4.`。

这是 [验证记录](../setup/mac-validation.md) 中的实测，不是每次换版本或换 prompt 都必须得到相同数字。保留最后一个输入位置重新求 logits 的边界与该现象相符。

## 不能由 demo 推出的结论

```text
玩具队列能结束       ≠ SRT allocator 无泄漏
最长前缀字符串相同   ≠ SWA/Mamba 状态都可复用
ZMQ 消息能通         ≠ OpenAI chat/SSE 全部正确
MPS 普通生成通过     ≠ CUDA Graph / 多卡 / PD 通过
```

验收时注明测了哪条路径，才能避免把局部成功扩大成整体结论。

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/managers/detokenizer_manager.py](../../python/sglang/srt/managers/detokenizer_manager.py) | 真实输出解码 |
| [python/sglang/srt/mem_cache/registry.py](../../python/sglang/srt/mem_cache/registry.py) | 当前缓存选择 |
