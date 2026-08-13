# 新人入门：30 分钟看懂一次生成

目标不是背类名，而是回答四个问题：请求在哪里、KV 放哪里、下一轮 token 从哪来、CPU 为什么可以晚一点处理结果。

## 先跑一个确定性例子

```bash
uv run my-sglang-generate \
  --input-ids 1,2 --token-ids 10,11,12,13 \
  --max-new-tokens 3 --overlap --trace
```

最后的标准输出是 `10,11,12`。trace 中只需先认出这些行（`B1/B2` 是连续 decode batch）：

```text
forward:sample:B0:[10]       # prefill B0 生成首 token 10
pipeline_process              # 先提交 10；请求才成为 RUNNING
forward:gather:B1:[10]       # decode B1 从 FutureMap 读取 10
forward:gather:B2:[11]       # B2 的 gather 已排在 B1 sampling/stash 之后
event:sync:B1.copy_done      # CPU 此时才等待并提交旧 decode B1
copy:d2h:B1:[11]             # CPU buffer 已有 11
```

## 用一个请求走完整生命周期

设 prompt 是 `[1,2]`，Fake 模型依次返回 `10,11,12`：

```text
请求 A

WAITING
  │ EXTEND: 输入 prompt [1,2]，生成 10
  ▼
RUNNING, output_ids=[10]
  │ DECODE: 输入 10，生成 11
  ▼
RUNNING, output_ids=[10,11]
  │ DECODE: 输入 11，生成 12
  ▼
FINISHED, output_ids=[10,11,12]
```

注意：一次 decode 的输入是“上一个已生成 token”，输出是“新的 token”。因此生成 `12` 的这次 forward 把 `11` 写入 KV；`12` 若已结束则不必再进入 KV。

## 再看 overlap 为什么成立

```text
首个 prefill 必须先 process，令请求进入 `RUNNING`。之后才有 overlap：

进入 Turn 3 前：
  result_queue = [B1]         # B1 是 decode，但 CPU 结果仍未提交
  A.output_ids = [10]

Turn 3：
  1. CPU enqueue B2（B2 的 gather 任务排在 B1 后）
  2. CPU 等 B1.copy_done；forward stream 依次执行 B1 sample/stash(11)、B2 gather(11)
  3. CPU 得到 B1 的 host token 11
  4. CPU: A.output_ids.append(11)，B1 出队
```

所以同一个 token 有两条用途：

| token 11 的去向 | 目的 | 是否需要等 CPU |
|---|---|---:|
| `FutureMap[A.row]` | 给 B2 当输入 | 否 |
| B1 的 host buffer | 追加到 `A.output_ids`、判断结束 | 是，等 `copy_done` |

## 建议阅读与断点顺序

| 顺序 | 读什么 | 只回答一个问题 |
|---:|---|---|
| 1 | `models.py` + `docs/data-structures.md` | 请求状态、row、KV 长度分别是什么？ |
| 2 | `schedule_batch.py` + `pools.py` | 一个逻辑 token 如何拿到 KV slot/page？ |
| 3 | `scheduler.py` | 没有 overlap 时如何 prefill、decode、finish？ |
| 4 | `runner.py` | Fake forward/copy stream 和 event 如何延迟执行？ |
| 5 | `overlap_scheduler.py` + `docs/overlap-pipeline.md` | 为什么 B1 先 launch、B0 后 process？ |
| 6 | `tests/test_overlap_scheduler.py` | 用断言验证前面每个结论。 |

第一轮只看单请求。理解后再增加一个变量：chunked prefill、radix cache、内存不足 retract、多个请求完成时的多算 token。
