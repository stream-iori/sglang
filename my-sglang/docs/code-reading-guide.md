# 代码阅读顺序

1. `models.py`：`Req` 的状态、KV 水位、`BatchForward` 快照。
2. `pools.py` 与 `schedule_batch.py`：row -> KV slot/page 的分配与回滚。
3. `scheduler.py`：同步 prefill/decode、chunk、radix 与 retract 基线。
4. `runner.py`：`FakeCudaStream`、`FakeCudaEvent`、`FakeCudaRunner`。
5. `overlap_scheduler.py`：FutureMap buffer、result FIFO、`launch B1 -> process B0`。
6. `tests/test_overlap_scheduler.py`：从 buffer、event 到完整 pipeline 的可执行证据。

重点区分：FutureMap 是下一轮 forward 的设备 token；`copy_done` 是 CPU 能读取 host token 的凭证；`output_ids` 只在 FIFO process 后更新。
