# 01 请求与调度：先看一条请求怎么走完

配套大图：[F1 对象关系、F2 请求状态](15-field-atlas.md)。

[返回全局地图](README.md) · 下一页：[02～03 KV 与工作单](02-memory-batch.md)

**请求跨轮保存上下文，调度器每轮挑一批请求交给模型。** 本页先用同步模式，无 cache、无 chunk；预设输出 10、11、12，生成上限为 3。

## 一条因果链，三个时间点

```text
输入 [7,8] --EXTEND--> 输出 10 --DECODE 输入 10--> 输出 11 --DECODE 输入 11--> 输出 12
  KV写 7,8                         KV写 10                          KV写 11
```

“KV 写 10”是为 token 10 计算并存储 K/V 向量，不是把整数 10 当作 K/V 存进去。

| 同步时刻 | 本轮输入 | 新输出 | output_ids | 已算出 K/V 的 token |
|---|---|---:|---|---|
| 刚入队 | 无 | 无 | `[]` | 无 |
| 完整 EXTEND 后 | `[7,8]` | 10 | `[10]` | `[7,8]` |
| 第一轮 DECODE 后 | `[10]` | 11 | `[10,11]` | `[7,8,10]` |
| 第二轮 DECODE 后、释放前 | `[11]` | 12 | `[10,11,12]` | `[7,8,10,11]` |

最后一行达到上限，随即结束并释放 active row/KV。token 12 不需要再输入模型。以上“最后输出尚无 KV”适用于同步观察点；overlap 要按[交接时序](05-overlap.md)判断。

## 状态描述请求，模式描述这一轮动作

```text
WAITING --完整 EXTEND--> RUNNING --DECODE--> RUNNING --停止条件--> FINISHED
   |
   +--中间 EXTEND--> PREFILLING --最后 EXTEND--> RUNNING
```

| 状态 | 说明 |
|---|---|
| WAITING | 等准入，新请求或撤回后待重建的请求 |
| PREFILLING | 有未完成 chunk，尚不能确认首 token |
| RUNNING | 上下文已补完，后续可以逐 token decode |
| FINISHED | 达到长度、EOS、stop token 或 abort |

完整 EXTEND 若立即达到停止条件，也可直接 FINISHED。RUNNING 不表示 GPU 此刻正在执行。

## 从一条请求扩展到多条：调度器每轮问什么

```text
结算 last_batch 的容器归属
          |
          v
有 waiting / chunked_req？ -> 尝试 prefill 准入
          |                     |
          |              ADMIT / CHUNK -> 准备 EXTEND
          |
没有可执行的 prefill batch -> 尝试 running_batch 的 DECODE
                                      |
                            准备 -> 执行 -> 提交 -> 交接
```

| 准入检查 | 为什么独立存在 |
|---|---|
| request row 是否有空位 | 每个活跃请求需要稳定地址表行 |
| KV 容量与 decode 预留 | 新 prompt 不能吃光后续生成要用的空间 |
| 本轮 prefill token 预算 | 限制本轮新增计算量，和总内存容量不同 |

`ADMIT` 是本轮处理全部剩余上下文，`CHUNK` 是处理一段，`DEFER` 是等以后，`ABORT` 是无法满足最小执行条件。当前 FCFS 候选遇到 DEFER，后面的候选不能越过它。

## 合上文档，复述这个例子

“已经输出 10，下一轮输入 10，写入 10 的 KV，再输出 11。”能明确每个动词的对象后，再读下一页。

深入：[Scheduler 一轮](../scheduler-kv-overview.md)；源码：[`MiniScheduler.step()`](../../src/my_sglang/scheduler.py)。
