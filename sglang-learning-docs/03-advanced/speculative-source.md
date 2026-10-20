# 投机解码：候选不是已经提交的输出

> 源码基准：上游 `16eb87c06f`；本地集成 `4f0e1cbea2`（2026-10-20）。本地日志增强与验证范围见总索引和 Mac 验证记录。

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

## 一轮投机的资源账本

| 阶段 | token 语义 | 存储动作 |
|---|---|---|
| draft | 候选，尚未提交 | 预留或写入候选状态 |
| target verify | 批量验证候选 | 读取候选窗口和已提交上下文 |
| accept | 得出可提交边界 | 正式追加输出，保留对应状态 |
| reject / 后续 extend | 修正并继续 | 清理无效分支，补齐下一轮所需状态 |

预留长度、候选个数、接受长度、正式输出长度是不同计数。统计接受率前先查分母和 bonus token 契约。

## 当前文件入口

DFlash 采用当前 `dflash_worker_v2.py` 路径；旧 `dflash_worker.py` 历史残留已清理。算法选择从 `spec_info.py` 进入，不按“名字看起来最像”的文件读。

统一 draft 池进一步支持 SWA/循环状态组合，状态接受与回收不只涉及 Full KV。target 和 draft 可有不同地址视角，禁止把 target 的裸索引无条件交给 draft。

## 性能推导

假设普通 target 一轮确认 1 个 token，投机一轮平均确认 K 个：投机是否划算，要比较 `(draft 时间 + verify 时间 + 额外状态管理) / K` 与普通单 token 时间。K 高也不保证快，padding、通信和候选模型大小都会影响分子。

实验固定 target、采样设置、输入输出长度与并发，先比输出正确性，再比较 TTFT、ITL、吞吐及各阶段耗时。本轮未执行投机 GPU 实验。

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/speculative/spec_info.py](../../python/sglang/srt/speculative/spec_info.py) | 算法和输入基类 |
| [python/sglang/srt/speculative/eagle_info.py](../../python/sglang/srt/speculative/eagle_info.py) | 候选/验证状态 |
| [python/sglang/srt/speculative/eagle_worker_v2.py](../../python/sglang/srt/speculative/eagle_worker_v2.py) | accept_lens 及 worker |
| [python/sglang/srt/model_executor/forward_batch_info.py](../../python/sglang/srt/model_executor/forward_batch_info.py) | 专用 ForwardMode |
