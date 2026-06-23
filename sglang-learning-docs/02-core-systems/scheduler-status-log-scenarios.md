# Scheduler 状态日志 11 个典型场景

> 目标：用 `scheduler.status` JSON 日志反推 Scheduler 正在做什么，重点看
> `running_batch`、`cur_batch`、`last_batch`、`waiting_queue`、
> `req_to_token_pool`、`token_to_kv_pool_allocator` 和 `radix_cache`。
>
> 上级索引：[Req 到 ScheduleBatch：状态流转导读](./request-batch-state-flow.md)

## 先读懂字段

一条 `scheduler.status` 日志通常长这样：

```json
{
  "running_rids": ["req-a"],
  "queued_rids": ["req-b"],
  "running_batch": {},
  "cur_batch": {},
  "last_batch": {},
  "waiting_queue": {},
  "req_to_token_pool": {},
  "token_to_kv_pool_allocator": {},
  "radix_cache": {}
}
```

核心字段含义：

| 字段 | 怎么读 |
|---|---|
| `running_batch` | Scheduler 当前维护的活跃 batch 状态快照，不是 waiting queue。 |
| `cur_batch` | 本轮实际提交/正在处理的 batch。纯 decode 时常与 `running_batch` 相同。 |
| `last_batch` | 最近一轮 forward 的 batch 引用，用于 EXTEND -> running 衔接；overlap 下也代表前一流水级。 |
| `waiting_queue` | 已进入 Scheduler、但还没被本轮 admission 成 EXTEND batch 的请求。 |
| `forward_mode=EXTEND` | prefill/extend，本轮输入多个 prompt/suffix token，通常写入多份 KV。 |
| `forward_mode=DECODE` | decode，本轮通常每个请求写入 1 个新 token 的 KV。 |
| `seq_lens` | 按请求对齐的当前总序列长度，不是“第几个 token”。 |
| `req_pool_indices` | 按请求对齐的 `ReqToTokenPool` 行号；`[2]` 表示第 2 行，不是占两个 slot。 |
| `out_cache_loc_len` | 本轮新分配/写入的 KV slot 数量，不是总 cache 长度。 |
| `active_rows` | 当前 running batch 中每个请求的 `req_pool_idx -> token_locs` 有界样本。 |
| `token_to_kv_pool_allocator.used_size` | KV pool 已用 slot 数。 |
| `radix_cache.evictable_size` | 可驱逐的 prefix cache token 数。 |
| `radix_cache.protected_size` | 被活跃请求引用、暂时不能驱逐的 token 数。 |

## 日志来源

场景 1-5 来自一次本地并发请求真实采样：

```text
/tmp/sglang_concurrent_log_check_20260624_001103/scheduler_status/StreamdeAir_0.log
```

采样组合：

```text
('EXTEND', size=1, waiting=0)
('EXTEND', size=1, waiting=5)
('EXTEND', size=5, waiting=0)
('DECODE', size=6, waiting=0)
('DECODE', size=3, waiting=0)
```

场景 6-11 是基于同一日志结构的教学构造样例，用于覆盖本次短跑没有稳定抓到
的边界情况。

## 场景 1：空服务后的健康检查 EXTEND

真实采样。服务刚启动后 health check 会触发一个极小请求：

```json
{
  "running_batch": {
    "size": 1,
    "forward_mode": "EXTEND",
    "seq_lens": [1],
    "req_pool_indices": [1],
    "out_cache_loc_len": 1
  },
  "waiting_queue": {
    "size": 0,
    "rids": []
  },
  "req_to_token_pool": {
    "active_rows": [
      {
        "rid": "HEALTH_CHECK_b00cf7dd99094e1dac6645dffdb968da",
        "req_pool_idx": 1,
        "seq_len": 1,
        "token_locs_head": [1],
        "token_locs_tail": [1]
      }
    ]
  },
  "token_to_kv_pool_allocator": {
    "available_size": 26393,
    "used_size": 1
  },
  "radix_cache": {
    "total_size": "(1, 0)",
    "evictable_size": 1,
    "protected_size": 0,
    "root_children": 1
  }
}
```

读法：

- 单个 health request 进入 EXTEND。
- `seq_lens=[1]`，本轮只写 1 个 KV slot。
- `used_size=1`，KV pool 中只有这个 token。

## 场景 2：一个请求 EXTEND，后面 5 个请求排队

真实采样。并发请求刚进来时，Scheduler 先让一个请求做 EXTEND，其余请求在
`waiting_queue` 中等待 admission：

```json
{
  "running_rids": ["d06aa973b7024657b45b9c77eef7ca1e"],
  "queued_rids": [
    "ffc2cd0fd72440ed93ed5807530bee8a",
    "cf3e1ec7f8d6421db70051fe9fda1ba3",
    "612bf3a5e4c549b986885f24d0d9752d",
    "4dbc3e116e46413685f527996d16d4f4",
    "8afd8a3c42264711a917e26bbaf6a1b1"
  ],
  "running_batch": {
    "size": 1,
    "forward_mode": "EXTEND",
    "seq_lens": [18],
    "req_pool_indices": [2],
    "out_cache_loc_len": 18
  },
  "cur_batch": {
    "size": 1,
    "forward_mode": "EXTEND",
    "seq_lens": [18],
    "req_pool_indices": [2],
    "out_cache_loc_len": 18
  },
  "last_batch": {
    "size": 1,
    "forward_mode": "EXTEND",
    "seq_lens": [18],
    "req_pool_indices": [2],
    "out_cache_loc_len": 18
  },
  "waiting_queue": {
    "size": 5
  },
  "req_to_token_pool": {
    "active_rows": [
      {
        "rid": "d06aa973b7024657b45b9c77eef7ca1e",
        "req_pool_idx": 2,
        "seq_len": 18,
        "token_locs_head": [2, 3, 4, 5, 6, 7, 8, 9],
        "token_locs_tail": [12, 13, 14, 15, 16, 17, 18, 19]
      }
    ]
  },
  "token_to_kv_pool_allocator": {
    "available_size": 26375,
    "used_size": 19
  },
  "radix_cache": {
    "total_size": "(19, 0)",
    "evictable_size": 1,
    "protected_size": 18,
    "root_children": 2
  }
}
```

读法：

- `waiting_queue.size=5` 表示有 5 个请求尚未被组成 EXTEND batch。
- waiting queue 里的请求通常还没有 `req_pool_idx`。
- 当前 running request 分到 `ReqToTokenPool` 第 2 行。
- `out_cache_loc_len=18`，这轮 EXTEND 为该请求写入 18 个 KV slot。

## 场景 3：多个请求一起 EXTEND

真实采样。后续 5 个请求被 admission 成一个 EXTEND batch：

```json
{
  "running_batch": {
    "size": 5,
    "forward_mode": "EXTEND",
    "seq_lens": [17, 16, 19, 18, 18],
    "req_pool_indices": [3, 4, 5, 6, 7],
    "out_cache_loc_len": 56
  },
  "cur_batch": {
    "size": 5,
    "forward_mode": "EXTEND",
    "seq_lens": [17, 16, 19, 18, 18],
    "req_pool_indices": [3, 4, 5, 6, 7],
    "out_cache_loc_len": 56
  },
  "last_batch": {
    "size": 5,
    "forward_mode": "EXTEND",
    "seq_lens": [17, 16, 19, 18, 18],
    "req_pool_indices": [3, 4, 5, 6, 7],
    "out_cache_loc_len": 56
  },
  "token_to_kv_pool_allocator": {
    "available_size": 26323,
    "used_size": 71
  },
  "radix_cache": {
    "total_size": "(71, 0)",
    "evictable_size": 1,
    "protected_size": 70,
    "root_children": 3
  }
}
```

按下标对齐：

| 请求下标 | `seq_lens` | `req_pool_indices` |
|---|---:|---:|
| req0 | 17 | 3 |
| req1 | 16 | 4 |
| req2 | 19 | 5 |
| req3 | 18 | 6 |
| req4 | 18 | 7 |

注意：`seq_lens` 总和是 88，但 `out_cache_loc_len=56`。这说明其中一部分
prompt token 命中了 prefix/radix cache，本轮只为未命中的 suffix 分配 KV。

## 场景 4：active_rows 显示多个请求共享 prefix KV

真实采样。与场景 3 是同一轮 EXTEND 的 `active_rows`：

```json
{
  "req_to_token_pool": {
    "active_rows": [
      {
        "rid": "ffc2cd0fd72440ed93ed5807530bee8a",
        "req_pool_idx": 3,
        "seq_len": 17,
        "token_locs_head": [20, 21, 22, 23, 24, 25, 26, 27],
        "token_locs_tail": [29, 30, 31, 32, 33, 34, 35, 36]
      },
      {
        "rid": "cf3e1ec7f8d6421db70051fe9fda1ba3",
        "req_pool_idx": 4,
        "seq_len": 16,
        "token_locs_head": [20, 21, 39, 40, 41, 42, 43, 44],
        "token_locs_tail": [45, 46, 47, 48, 49, 50, 51, 52]
      },
      {
        "rid": "612bf3a5e4c549b986885f24d0d9752d",
        "req_pool_idx": 5,
        "seq_len": 19,
        "token_locs_head": [20, 21, 55, 56, 57, 58, 59, 60],
        "token_locs_tail": [64, 65, 66, 67, 68, 69, 70, 71]
      }
    ]
  }
}
```

读法：

- 多个请求的 `token_locs_head` 都以 `[20, 21, ...]` 开头。
- 这表示它们共享了同一段 prefix KV slot。
- 后面分叉到不同 KV slot，表示各自 suffix 不同。

## 场景 5：6 个请求一起 DECODE

真实采样。所有请求进入 continuous batching decode：

```json
{
  "running_batch": {
    "size": 6,
    "forward_mode": "DECODE",
    "seq_lens": [19, 18, 17, 20, 19, 19],
    "req_pool_indices": [2, 3, 4, 5, 6, 7],
    "out_cache_loc_len": 6
  },
  "cur_batch": {
    "size": 6,
    "forward_mode": "DECODE",
    "seq_lens": [19, 18, 17, 20, 19, 19],
    "req_pool_indices": [2, 3, 4, 5, 6, 7],
    "out_cache_loc_len": 6
  },
  "last_batch": {
    "size": 6,
    "forward_mode": "DECODE",
    "seq_lens": [19, 18, 17, 20, 19, 19],
    "req_pool_indices": [2, 3, 4, 5, 6, 7],
    "out_cache_loc_len": 6
  },
  "token_to_kv_pool_allocator": {
    "available_size": 26317,
    "used_size": 77
  },
  "radix_cache": {
    "total_size": "(71, 0)",
    "evictable_size": 1,
    "protected_size": 70,
    "root_children": 3
  }
}
```

读法：

- `forward_mode=DECODE` 表示这轮逐 token 生成。
- `size=6` 且 `out_cache_loc_len=6`，说明 6 个请求各写 1 个新 KV slot。
- `seq_lens` 是每个请求当前总长度，不是“第几个 token”。

## 场景 6：DECODE active_rows 的 tail 体现新 token

真实采样。与场景 5 对应：

```json
{
  "req_to_token_pool": {
    "active_rows": [
      {
        "rid": "d06aa973b7024657b45b9c77eef7ca1e",
        "req_pool_idx": 2,
        "seq_len": 19,
        "token_locs_head": [2, 3, 4, 5, 6, 7, 8, 9],
        "token_locs_tail": [13, 14, 15, 16, 17, 18, 19, 76]
      },
      {
        "rid": "ffc2cd0fd72440ed93ed5807530bee8a",
        "req_pool_idx": 3,
        "seq_len": 18,
        "token_locs_head": [20, 21, 22, 23, 24, 25, 26, 27],
        "token_locs_tail": [30, 31, 32, 33, 34, 35, 36, 77]
      }
    ]
  }
}
```

读法：

- decode 阶段看 `token_locs_tail` 很关键。
- 第一个请求 tail 最后是 `76`，第二个请求 tail 最后是 `77`。
- 这些就是本轮 decode 新追加的 KV slot。

## 场景 7：部分请求完成，DECODE batch 变小

真实采样。6 个请求中一部分完成后，running batch 缩到 3 个：

```json
{
  "running_batch": {
    "size": 3,
    "forward_mode": "DECODE",
    "seq_lens": [19, 18, 21],
    "req_pool_indices": [2, 3, 5],
    "out_cache_loc_len": 3
  },
  "cur_batch": {
    "size": 3,
    "forward_mode": "DECODE",
    "seq_lens": [19, 18, 21],
    "req_pool_indices": [2, 3, 5],
    "out_cache_loc_len": 3
  },
  "token_to_kv_pool_allocator": {
    "used_size": 80
  },
  "radix_cache": {
    "total_size": "(74, 0)"
  }
}
```

读法：

- `size` 从 6 变成 3，说明 3 个请求已 finish 并被过滤。
- `out_cache_loc_len=3`，剩下 3 个请求各写 1 个新 KV slot。

## 场景 8：DECODE 正在跑，同时新请求排队

教学构造样例。真实服务常见，但本次短跑没有稳定抓到这一刻：

```json
{
  "running_batch": {
    "size": 4,
    "forward_mode": "DECODE",
    "seq_lens": [128, 97, 66, 41],
    "req_pool_indices": [2, 3, 4, 5],
    "out_cache_loc_len": 4
  },
  "waiting_queue": {
    "size": 2,
    "rids": ["new-prefill-a", "new-prefill-b"]
  },
  "req_to_token_pool": {
    "active_rows": [
      {
        "rid": "decode-a",
        "req_pool_idx": 2,
        "seq_len": 128,
        "token_locs_head": [10, 11, 12, 13],
        "token_locs_tail": [300, 301, 302, 800]
      }
    ]
  },
  "token_to_kv_pool_allocator": {
    "available_size": 25000,
    "used_size": 1394
  },
  "radix_cache": {
    "total_size": "(1200, 0)",
    "evictable_size": 200,
    "protected_size": 1000
  }
}
```

读法：

- 老请求正在 decode，新请求先进 `waiting_queue`。
- 后续 scheduler 可能继续 decode，也可能 admission 新请求做 EXTEND。

## 场景 9：大量 prefix cache 命中

教学构造样例。重点是 `out_cache_loc_len` 远小于 `seq_lens` 总和：

```json
{
  "running_batch": {
    "size": 3,
    "forward_mode": "EXTEND",
    "seq_lens": [100, 101, 102],
    "req_pool_indices": [8, 9, 10],
    "out_cache_loc_len": 9
  },
  "req_to_token_pool": {
    "active_rows": [
      {
        "rid": "cache-hit-a",
        "req_pool_idx": 8,
        "seq_len": 100,
        "token_locs_head": [1000, 1001, 1002, 1003],
        "token_locs_tail": [1096, 1097, 5001, 5002]
      }
    ]
  },
  "token_to_kv_pool_allocator": {
    "available_size": 24991,
    "used_size": 1403
  },
  "radix_cache": {
    "total_size": "(1394, 0)",
    "evictable_size": 900,
    "protected_size": 494
  }
}
```

读法：

- 三个请求总长度 303，但本轮只新写 9 个 KV slot。
- 大部分 prompt token 都通过 radix cache 复用了已有 KV。

## 场景 10：KV pool 紧张，但可驱逐 cache 很多

教学构造样例：

```json
{
  "running_batch": {
    "size": 2,
    "forward_mode": "DECODE",
    "seq_lens": [2048, 1536],
    "req_pool_indices": [11, 12],
    "out_cache_loc_len": 2
  },
  "token_to_kv_pool_allocator": {
    "available_size": 120,
    "used_size": 26274
  },
  "radix_cache": {
    "total_size": "(21000, 0)",
    "evictable_size": 18000,
    "protected_size": 3000
  }
}
```

读法：

- KV pool 接近满，`available_size` 很低。
- `evictable_size` 很大，说明可以通过驱逐 prefix cache 腾出空间。
- 这是“紧张但还能回收”的状态。

## 场景 11：KV pool 紧张，且 protected 太多

教学构造样例。这个比场景 10 更危险：

```json
{
  "running_batch": {
    "size": 8,
    "forward_mode": "DECODE",
    "seq_lens": [4096, 3900, 3800, 3700, 3600, 3500, 3400, 3300],
    "req_pool_indices": [20, 21, 22, 23, 24, 25, 26, 27],
    "out_cache_loc_len": 8
  },
  "token_to_kv_pool_allocator": {
    "available_size": 64,
    "used_size": 26330
  },
  "radix_cache": {
    "total_size": "(26000, 0)",
    "evictable_size": 200,
    "protected_size": 25800
  }
}
```

读法：

- KV pool 几乎满。
- `protected_size` 很大，表示大部分 cache 正被活跃请求引用，不能驱逐。
- 后续更可能触发 retract、等待、降低 batch 或更保守的调度。

## 常见误读修正

| 误读 | 正确读法 |
|---|---|
| `running_batch` 是待调度 batch | 它是当前活跃/运行 batch 状态快照；待调度看 `waiting_queue`。 |
| `req_pool_indices=[2]` 表示占用 2 个 slot | 表示第一个请求使用 `ReqToTokenPool` 第 2 行。 |
| `seq_lens=[8]` 表示第 8 个 token | 表示该请求当前总序列长度为 8。 |
| `out_cache_loc_len` 是输出 cache 总长度 | 表示本轮新分配/写入的 KV slot 数量。 |
| `running_batch/cur_batch/last_batch` 相同就一定是 normal loop | 不能单凭这个判断；overlap/MLX 某些采样时刻也可能三者相同。 |

## 读日志检查清单

1. 先看 `forward_mode`：本轮是 `EXTEND` 还是 `DECODE`。
2. 再看 `running_batch.size` 和 `waiting_queue.size`：是在跑、在排队，还是两者都有。
3. 用下标对齐 `seq_lens` 和 `req_pool_indices`。
4. 看 `out_cache_loc_len`：EXTEND 写多少 suffix KV；DECODE 是否等于 batch size。
5. 看 `active_rows.token_locs_head/tail`：是否共享 prefix，decode 是否追加新 slot。
6. 看 `token_to_kv_pool_allocator.available_size/used_size`：KV pool 是否紧张。
7. 看 `radix_cache.evictable_size/protected_size`：紧张时能不能驱逐 cache。
