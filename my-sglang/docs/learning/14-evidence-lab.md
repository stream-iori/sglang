# 贯穿实验（三）：从现象回到证据

[返回全局地图](README.md) · 上一页：[共享有限 KV](13-shared-kv.md)

**先写观察点，再读状态，最后下结论。** 下面前三项可在 CPU 教学运行时重现；真实 GPU 显存与性能必须另取设备和服务证据。

## 图 S20：固定的判断路径

```text
现象 -> 我在哪个时刻观察？ -> 哪份记录能回答？ -> 对照源码 -> 有边界的结论
          step返回后？          地址表 / trace       真正读写点
          launch之后？          cache / output      释放与提交点
```

在 `my-sglang` 目录执行。每行 JSON 是一个 step/turn 的快照；`cache` 列表记录压缩树边的局部 key、slot 和 lock，不能将每条边当作从根起的完整 prefix。

## 实验 1：命中缓存，为什么还执行 EXTEND？

```bash
uv run python examples/runtime_walkthrough.py shared
```

| 看什么 | 实跑证据 | 结论 |
|---|---|---|
| step3 B 的 input | `[11]` | 只计算未命中的 suffix |
| B mapping | `[2,3,6]` | 前两个地址与 A 共用，新输入写 slot6 |
| cache `[7,8]` 的 lock | 2 | A、B 都在借用 |

命中 prefix 减少计算范围，不保证整次调用消失。完整命中的特殊情况继续对照 tiny 的 hidden 缓存与[标准匹配上限](06-standard-execution.md)。

定位：[`_attach_new_request()`](../../src/my_sglang/scheduler.py) → [`prepare_for_extend()`](../../src/my_sglang/schedule_batch.py)。

## 实验 2：请求都结束了，为什么池还满？

同一命令观察 step7：A、B 均 FINISHED、row=null，mapped_tokens=0，但 allocated_tokens=8，cache_evictable_tokens=8。

```text
active 请求释放 -> 完整页转交 radix -> allocator 仍记为占用
                                          |
                           C 后续入队 -> 淘汰无锁页 -> 复用
```

这证明的是 **教学池的所有权交接**，不能直接证明真实服务 `nvidia-smi` 中任何一块显存的用途。真实服务还要区分预分配 KV pool、模型权重、执行工作区与框架缓存；释放请求映射不要求把整块设备分配立即归还驱动。

定位：[`_cache_slots_to_keep_on_release()` / `_release_active_memory()`](../../src/my_sglang/scheduler.py)。

## 实验 3：overlap 快照里的输入落后一轮，是 bug 吗？

```bash
uv run python examples/runtime_walkthrough.py overlap
```

| 看什么 | 证据 |
|---|---|
| turn3 的新 B2 快照 | input 为 `[10]` |
| 最后一行 trace | `forward:gather:B2:[11]` |
| 最终确认输出 | `[10,11,12]` |

此处 CPU 快照不是实际 decode 输入来源，执行时从 FutureMap 读取。结论只适用于当前 overlap relay 路径，不能拿它替任何模型输入异常开脱。

定位：[FakeCudaRunner 的 gather](../../src/my_sglang/runner.py)、[MiniOverlapScheduler](../../src/my_sglang/overlap_scheduler.py)。

## 实验 4：同步与 overlap 是否正确释放、恢复？

先跑两个状态案例，再运行已有针对性测试：

```bash
uv run python examples/runtime_walkthrough.py sync
uv run --extra test pytest tests/test_overlap_scheduler.py tests/test_pools.py -q
```

脚本检查最终输出与 row 回收；已有测试还验证 event 只推进必要前缀、FutureMap 消费检查、失败后保留确认输出并重新排队、尾页复用等。它们验证教学语义，不证明标准 SRT 对任意设备错误都能恢复。

## 真实 GPU 观察任务：首输出慢在哪里？

本批未运行真实 GPU 服务，下表是后续实验设计，不是测量结果。

| 先收集 | 再比较 | 可以回答 |
|---|---|---|
| 同一 rid 的客户端发送、服务接收、首输出时间 | 客户端与服务侧时间边界 | 等待是否包含网络/输出缓冲 |
| waiting、prefill 的时间与输入长度 | 相同负载下的排队与计算占比 | 慢在准入还是 prefill |
| cache 命中、实际 extend token 数 | 相同 prompt 的冷/热请求 | 缓存是否减少实际计算范围 |
| GPU 算子、通信与 CPU 提交时间线 | 相同并发下的 graph/普通执行 | 执行瓶颈在什么位置 |

只拿一条客户端耗时无法区分这些阶段。先按[性能观察](11-tp-performance.md)固定模型、请求长度、并发、cache 冷热和计时单位，再做比较。

## 本批验证记录

本批已运行三个脚本入口，均正常结束且断言通过；`tests/test_overlap_scheduler.py` 与 `tests/test_pools.py` 共 10 项测试通过。未进行真实 GPU 性能测量。

配套脚本有 `sync`、`overlap`、`shared` 三个入口。数值是当前教学代码实跑快照；未来实现改变时重新运行，并以新的因果证据更新表格，不应为了保留旧数字修改调度行为。
