# 03～05 overlap：先交输入，再确认旧结果

配套大图：[F5 异步交接与三处记录](15-field-atlas.md)。

[返回全局地图](README.md) · 上一页：[分支](04-branches.md)

下一页：[标准 SRT 的真实执行链](06-standard-execution.md)。本页故障恢复描述的是教学实现；标准设备异常不保证可以自动重排，详见[输出与异常边界](08-standard-output.md)。

**overlap 错开 CPU 工作与执行侧工作，但同一个请求的 token 依赖仍然串行。** 本页 B0、B1 都是稳定阶段的 DECODE batch，不是首次 prefill。

## 同步与 overlap 只先比较这一段

```text
同步：    执行 B0 -> CPU 确认 B0 -> 准备并执行 B1

overlap：CPU    enqueue B1 -> wait B0.copy_done -> process B0
         执行侧 B0 算出 10 -> stash FM[row]=10 -> B1 gather 10 -> 算后续 token
         copy   B0 结果复制完成 ----------------> CPU 才能确认
```

上图表达依赖，不表示所有动作同时完成。Fake CUDA 的等待只推进 event 必需的队列前缀：等待 B0.copy_done 不会顺便执行 B1 forward；真实 CUDA 可自主推进已提交工作。

## 三处 token 记录，分别回答三个问题

| 记录 | 内容 | 可信边界 |
|---|---|---|
| FutureMap[row] | 最新生产、供下次 decode 消费的 token | 执行侧数据依赖已满足，不代表 CPU 确认 |
| result_queue 中的 job/result | batch 快照、异步结果与完成事件 | 排队不代表 token 已算出或已复制 |
| Req.output_ids | CPU 按 FIFO 确认的输出历史 | 故障恢复必须保留 |

假设 B0 已 stash 10、B1 尚未 gather，CPU 尚未 process：FutureMap 有 10；B0 的执行结果已有 10，但 host 副本是否就绪要看 copy_done；output_ids 尚无这个 10。不能仅凭“result 已入队”就说 host 已拿到 token。

```text
稳定 row=3                     逻辑位置
FutureMap[3] = 10              req_to_token[3,pos] = KV slot
      |                                 |
      v                                 v
下次 decode 的输入编号          本轮写 K/V 与读历史的地址
```

`stash` 会写 token 并令 valid=True；`gather` 读取后令 valid=False。valid 是教学/调试的“一次生产一次消费”检查，防止把旧值当新结果；False 不会阻止 stash 覆盖。标准生产实现不一定保留此检查。

## 这里必须纠正“committed 等于 forward 完成”

| 路径 | committed 前进时刻 | 尚未保证的事 |
|---|---|---|
| 同步 | runner 成功返回之后 | 新采样 token 本身尚无 KV |
| overlap | run_batch_async 成功返回，即 launch 成功后 | forward 完成、D2H 完成、CPU output 提交均不能由此推出 |

所以 overlap 中 committed 表示调度器接受了这次 KV 推进；若异步执行后来失败，要走整个 pipeline 的恢复，不能只把它当同步未提交分配失败。

## 停止与故障为什么必须晚点回收

```text
先 launch B1，后 process B0
                       |
                  发现 B0 是 EOS
                       |
           标记 FINISHED，但 B1 仍引用 row/KV
                       |
        等 B1 安全结算，丢弃其额外 token -> 最后释放资源
```

否则新请求复用同一 row 时，旧 B1 可能读写到新请求的资源。这里的多算是流水中提前提交的后继工作，不是另一套 speculative decoding 算法。

| 故障范围 | 必须做什么 |
|---|---|
| 同步 runner 失败 | 回滚未提交映射与独占新页，allocated 拉回 committed |
| overlap pipeline 失败 | discard 在途结果，清队列和受影响 FutureMap，释放受影响请求物理状态，未完成请求重新排队 |
| 已有确认输出 | 保留 output_ids，以 prompt + 确认输出重建；未确认结果可以丢弃 |

自检：B1 已入队，但 CPU 还没确认 B0，为什么不违反自回归？B1 执行时仍要等 B0 生产 token；提前的是提交工作，依赖并未消失。

深入：[event 与完整时序](../overlap-pipeline.md)、[标准 SRT 连续 prefill](../prefill-overlap.md)。源码：[`pipeline_step / FutureMap / recovery`](../../src/my_sglang/overlap_scheduler.py)。
