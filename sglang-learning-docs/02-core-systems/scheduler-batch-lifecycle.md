# Scheduler：每轮先选，再算，再处理结果

> 源码基准：learning `2aa70e3eb4`，已合入 origin/main `6fc8d9da32`（2026-10-10）。

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

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/managers/scheduler.py](../../python/sglang/srt/managers/scheduler.py) | normal/overlap 和调度计划 |
| [python/sglang/srt/managers/scheduler_components/batch_result_processor.py](../../python/sglang/srt/managers/scheduler_components/batch_result_processor.py) | 结果处理组件 |
| [python/sglang/srt/managers/scheduler_components/request_receiver.py](../../python/sglang/srt/managers/scheduler_components/request_receiver.py) | 请求接收组件 |
