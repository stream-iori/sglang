# 贯穿实验（一）：一条请求的完整状态账本

[返回全局地图](README.md) · 上一页：[TP 与性能](11-tp-performance.md) · 下一页：[共享有限 KV](13-shared-kv.md)

**把“这一轮输入”“确认输出”“KV 边界”放在同一时间点比较。** 本页数值来自配套脚本实跑，不是抽象时序推测。

在 `my-sglang` 目录执行：

```bash
uv run python examples/runtime_walkthrough.py sync
uv run python examples/runtime_walkthrough.py overlap
```

配置：prompt=`[7,8]`，max_new_tokens=3，page_size=2，可用 token slots=8，关闭 radix cache。脚本 runner 依次提供 10、11、12、13；它模拟调度与异步依赖，不计算真实 K/V 向量。

## 图 S15：同步与 overlap 并排

```text
同步                                overlap
执行 EXTEND -> 确认 10               turn1: enqueue B0 EXTEND
执行 DECODE(10) -> 确认 11           turn2: process B0 -> enqueue B1
执行 DECODE(11) -> 确认 12 -> 释放   turn3: enqueue B2 -> process B1
                                    turn4: enqueue B3 -> process B2
                                           发现结束 -> 结算 B3、丢弃 13 -> 释放
```

B0 是 prefill；B1、B2、B3 是 decode。稳定 decode 的“先提交后处理”从 turn3 开始体现。

## 同步：每个 step 返回之后

| step | 本轮输入 → 输出 | 状态 | output_ids | row / 映射 | allocated / committed | 池占用 slots |
|---:|---|---|---|---|---|---:|
| 1 | `[7,8] → 10` | RUNNING | `[10]` | 0 / `[2,3]` | 2 / 2 | 2 |
| 2 | `[10] → 11` | RUNNING | `[10,11]` | 0 / `[2,3,4]` | 3 / 3 | 4 |
| 3 | `[11] → 12` | FINISHED | `[10,11,12]` | 无 / 空 | 0 / 0 | 0 |

step2 请求映射了 3 个位置，却占 4 个 slots，因为按整页分配。step3 在释放前，已处理的逻辑 token 为 `[7,8,10,11]`；表中读取的是释放后的请求水位，所以为 0。

## overlap：每个 pipeline_step 返回之后

| turn | 新提交 / 本轮处理 | 状态 | output_ids | allocated / committed | FutureMap 有效值 | 队列 |
|---:|---|---|---|---|---|---|
| 1 | B0 / 无 | WAITING | `[]` | 2 / 2 | 无 | B0 |
| 2 | B1 / B0 | RUNNING | `[10]` | 3 / 3 | row0=10 | B1 |
| 3 | B2 / B1 | RUNNING | `[10,11]` | 4 / 4 | row0=11 | B2 |
| 4 | B3 / B2、B3 | FINISHED | `[10,11,12]` | 0 / 0 | 无 | 空 |

turn1 仍标 WAITING，但 row 已分配且 B0 在途：教学状态枚举只描述 CPU 尚未处理结果的阶段，不能据此断言资源为空。committed 已推进也不代表 Fake forward 已执行。

## 图 S16：一处最容易看错的日志

```text
turn3 准备 B2 时：CPU output_ids 只有 [10]
     -> ForwardBatch 输入快照可能仍显示 [10]
     -> B2 入队，尚未执行

随后 B1 执行：gather 10 -> sample 11 -> stash 11
再到 B2 执行：gather 11 -> sample 12
```

**overlap decode 的实际输入看 `forward:gather`，不能只看准备时的 CPU input_ids 快照。** 配套脚本最后一行 trace 包含：

```text
forward:gather:B1:[10]
forward:gather:B2:[11]
forward:gather:B3:[12]
```

内部多算的 13 没有进入 output_ids。结算完后 active row 数为 0；这是脚本实际断言的结果。

## 自检

| 问题 | 答案 |
|---|---|
| turn1 committed=2，为什么 output 为空？ | launch 已被接受，CPU 尚未处理异步结果。 |
| turn3 快照显示 10，是否证明重复输入旧 token？ | 不证明；实际设备侧输入由 FutureMap gather 决定。 |
| 结束后 committed=0，是否表示模型从未计算？ | 不是；这是释放后重置的水位。 |

代码：[实验脚本](../../examples/runtime_walkthrough.py)、[overlap 调度器](../../src/my_sglang/overlap_scheduler.py)、[runner](../../src/my_sglang/runner.py)。
