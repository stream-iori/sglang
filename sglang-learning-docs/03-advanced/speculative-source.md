# 投机解码：候选不是已经提交的输出

> 源码基准：learning `2aa70e3eb4`，已合入 origin/main `6fc8d9da32`（2026-10-10）。

小模型/候选机制先提出一段 token，大模型批量验证。只有接受的部分才能成为正式输出和已提交状态。

```text
已提交上下文：[A B]
draft 候选：        [c d e]
target verify：      ✓ ✓ ×
提交：              [c d] + 验证路径决定的后续 token
丢弃/回收：                  e 及未接受分支的暂存状态
```

示意图不对应每种算法的精确采样规则。保持目标分布的接受/修正算法与 greedy 验证不同，不能概括成简单字符串比对。

## 当前读码路径

| 层 | 文件/数据 | 问题 |
|---|---|---|
| 算法选择 | SpeculativeAlgorithm | 当前配置走哪个 worker |
| 候选与验证输入 | SpecInput、EagleDraftInput、EagleVerifyInput | 张量和树状候选怎么表示 |
| 执行 | eagle_worker_v2.py 等 | draft、target、draft extend 的先后关系 |
| 结果 | accept_lens、num_correct_drafts | 正式新增几个，正确 draft 有几个 |
| 状态 | allocation、Req KV 信息 | 预留、提交、释放怎样区别 |

EAGLE v2 的相关结果处理明确说明：`accept_lens` 包含 bonus token，`num_correct_drafts = accept_lens - 1`。不要把 accept_lens 当成纯 draft 命中数；其他算法仍需检查自身契约。

`TARGET_VERIFY` 和 `DRAFT_EXTEND_V2` 是 ForwardMode 中的专用模式。对应 Graph 也要满足专门的 shape 和 metadata 条件。

## 排查一次错误

| 现象 | 先看证据 |
|---|---|
| 输出重复或漏 token | 接受结果、输出追加边界、增量解码位置 |
| KV 长度比输出长 | 预分配与正式提交长度，是否尚在验证 |
| 接受率高却吞吐低 | draft/verify 时间、通信、padding、每轮正式输出数 |
| 状态回收后报错 | 分支槽位生命周期、锁、copy-on-write |

Mac 本地脚本只验证普通生成，未启用投机。先用上述链路读懂，再在受支持模型/设备上做性能实验。

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/speculative/spec_info.py](../../python/sglang/srt/speculative/spec_info.py) | 算法和输入基类 |
| [python/sglang/srt/speculative/eagle_info.py](../../python/sglang/srt/speculative/eagle_info.py) | 候选/验证状态 |
| [python/sglang/srt/speculative/eagle_worker_v2.py](../../python/sglang/srt/speculative/eagle_worker_v2.py) | accept_lens 及 worker |
| [python/sglang/srt/model_executor/forward_batch_info.py](../../python/sglang/srt/model_executor/forward_batch_info.py) | 专用 ForwardMode |
