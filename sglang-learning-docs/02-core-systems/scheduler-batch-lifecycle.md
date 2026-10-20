# Scheduler：每轮先选，再算，再处理结果

> 源码基准：上游 `16eb87c06f`；本地集成 `4f0e1cbea2`（2026-10-20）。本地日志增强与验证范围见总索引和 Mac 验证记录。

调度的第一性原理：有限的计算时间和 KV 容量，要分配给已经到来的请求。

## 普通循环与 overlap

```text
普通循环时间 →
CPU   接收/选批/准备 ───── 等设备 ───── 处理结果 ── 下一轮准备
设备                  forward + sample

overlap 时间 →（概念图）
CPU   准备 batch N ── 准备 batch N+1 ── 处理 N 结果
设备                 执行 N ───────── 执行 N+1
```

overlap 会延后结果处理，并维护批次记录/未来结果；不是两个随意修改同一个 Req 的循环。Mac 启动脚本关闭它，先学普通循环。

## 每一步在解决什么

| 步骤 | 输入 | 输出/副作用 |
|---|---|---|
| ingest_requests | IPC 消息 | 新 Req、控制请求 |
| get_next_batch_to_run | running_batch、last_batch、等待队列 | plan，决定本轮 batch |
| prepare_for_extend/decode | Req 与 pool | 输入 ID、长度、KV 写地址 |
| run_batch | ScheduleBatch | worker 结果 |
| process_batch_result | 结果与请求 | 追加 ID、检查结束、输出、缓存/释放 |
| on_idle | 没有可执行批次 | 自检、idle 处理 |

## 例子：两个请求陆续加入

| 轮次 | 到达情况 | 可能执行 | 状态变化 |
|---|---|---|---|
| 1 | A 到达 | A 的 EXTEND | A 获得首个输出，进入 decode 集合 |
| 2 | 无新增 | A 的 DECODE | A 输出继续增长 |
| 3 | B 到达 | B 的 EXTEND 或兼容的混合批 | A 的已有 KV 保留，B 建立 KV |
| 4 | A/B 未结束 | 两者 DECODE | 一个结束后只移除它 |

表是说明性场景，不保证所有配置下都如此排批。调度策略、chunk、预算、优先级和高级模式都会影响选择。

## 读码练习

在 `get_next_batch_to_run` 找到上一轮 extend 的过滤、chunk 的排除和 merge。观察返回值的类型，再向下追 `batch_to_run`，避免沿用“直接返回 ScheduleBatch”的旧假设。

## 调度计划中的两条线

```text
last_batch：上一轮做了什么 ──过滤结束/chunk──┐
                                         ├─ running_batch：后续 decode 集合
waiting_queue：新请求 ──匹配/预算/admission─┘
                                         ↓
                              NextBatchPlan.batch_to_run
```

先读 `get_next_batch_to_run` 中对上一轮 extend 的合并，再读新 prefill 的选取和 decode 更新。`running_batch` 不是“本轮一定正在设备执行”的同义词。

## overlap 需要保护什么

| 问题 | 当前机制线索 | 错误理解 |
|---|---|---|
| 结果还没拷回 CPU | result_queue、future indices、结果处理时机 | forward 返回就能同步读完所有输出 |
| 下一轮写共享输入时，上一轮仍在读 | _apply_war_barrier、event/stream 依赖 | CPU 发起下一轮就一定安全 |
| 不兼容 batch 需要同步边界 | is_disable_overlap_for_batch | overlap 对所有批次无条件开启 |
| 统一内存的延后动作 | flush_opportunistic、deferred release | Python 调用了 free 就能立刻复用底层字节 |

这些保护围绕读写时序和资源生命周期。Mac 普通循环可以学习状态转移，但不能用它的日志证明 CUDA overlap 的竞争条件已验证。

## 结束、回退和暂停

| 事件 | 需要追的状态 |
|---|---|
| EOS / 长度上限 | finished_reason、输出结束、后续过滤、缓存/释放 |
| 容量不足导致 retract | 活跃请求退回等待、KV 保留或回收规则、后续恢复输入 |
| 引擎暂停 | ingest 与状态记录仍需区分，暂停期间不能伪造 forward |
| PD 有延后释放 | decode 空闲判断还要考虑未完成传输/释放工作 |

## 动手

启动带状态日志的 Mac 服务，发送两个独立请求；记录 `running_batch.forward_mode`、rid 和 `last_batch`。日志按间隔采样，可能跳过极短的中间状态，因此“某轮没出现”不能证明调度器没执行。

验收答案：能解释为什么 normal 模式也有 last_batch，为什么 overlap 队列记录必须保留与结果对应的 batch，以及 CPU 准备和设备计算并行时需要什么先后约束。

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/managers/scheduler.py](../../python/sglang/srt/managers/scheduler.py) | normal/overlap 和调度计划 |
| [python/sglang/srt/managers/scheduler_components/batch_result_processor.py](../../python/sglang/srt/managers/scheduler_components/batch_result_processor.py) | 结果处理组件 |
| [python/sglang/srt/managers/scheduler_components/request_receiver.py](../../python/sglang/srt/managers/scheduler_components/request_receiver.py) | 请求接收组件 |
