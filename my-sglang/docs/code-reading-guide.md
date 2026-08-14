# 代码阅读顺序

如果尚未理解 prefill/decode 或 `FutureMap`，先完成 [新人入门](newcomer-guide.md)。本页假设你已经知道“每轮生成一个新 token”。

1. `models.py`：`Req` 的状态、KV 水位、`ForwardBatch` 快照。
2. `pools.py` 与 `schedule_batch.py`：row -> KV slot/page 的分配与回滚。
3. `scheduler.py`：同步 prefill/decode、chunk、radix 与 retract 基线。
4. `runner.py`：`FakeCudaStream`、`FakeCudaEvent`、`FakeCudaRunner`。
5. `overlap_scheduler.py`：FutureMap buffer、result FIFO、`launch B1 -> process B0`。
6. `tests/test_overlap_scheduler.py`：从 buffer、event 到完整 pipeline 的可执行证据。

## 每层读完后应能回答什么

| 文件 | 读完后的检查问题 |
|---|---|
| `models.py` | 为什么 `output_ids` 和 `kv_committed_len` 不是同一件事？ |
| `schedule_batch.py` | 为什么 decode 每个请求只申请一个新的 KV 位置？ |
| `scheduler.py` | 为什么 prefill 优先、什么时候 retract？ |
| `runner.py` | 为什么 `run_batch_async()` 返回后，token 还不能被 CPU 读取？ |
| `overlap_scheduler.py` | 为什么 `pipeline_step()` 先 launch B1 再 resolve B0？ |

重点区分：FutureMap 是下一轮 forward 的设备 token；`copy_done` 是 CPU 能读取 host token 的凭证；`output_ids` 只在 FIFO process 后更新。

读完基础链路后，再看 [标准 SRT 的连续 prefill overlap](prefill-overlap.md)。它把
同一个 `launch current -> process previous` 不变量扩展到多个在途 prompt chunk，
需要额外维护 `inflight_middle_chunks`，不只是放开一道 barrier。
