# 练习答案与验证边界

> 源码基准：上游 `16eb87c06f`；本地集成 `4f0e1cbea2`（2026-10-20）。本地日志增强与验证范围见总索引和 Mac 验证记录。

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

## 新增练习答案

| 编号 | 答案要点 |
|---|---|
| 5 | A 新算 2，B 新算 3；packed 偏移与序列 position 分开；物理槽位由分配器决定 |
| 6 | 对 A/B 分别比较输出 ID；不能将 A 与 B 相互比较，也不能用文本看起来不同当隔离证据 |
| 7 | status 是采样；外部 cur_batch 读内部 cur_batch_for_debug；只有当次 run_id 算本次证据 |
| 8 | publish 投影当前配置；override 改有效 bag 并记录来源；use_context 切进程当前引用，要求串行 |
| 9 | buffer info 描述持久布局事实；容量和层路由由装配层提供，Host 池负责相应镜像/传输 |
| 10 | DeepSeek V4 replay tail；先让 workspace 达到稳定范围，避免 Graph 引用旧地址 |

若实测与历史 token 数不同，先检查输入、tokenizer、模板、配置和版本，再判断错误。验收关注机制和当次记录，不把历史 197 当作必须通过的常量。

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/managers/detokenizer_manager.py](../../python/sglang/srt/managers/detokenizer_manager.py) | 真实输出解码 |
| [python/sglang/srt/mem_cache/registry.py](../../python/sglang/srt/mem_cache/registry.py) | 当前缓存选择 |
