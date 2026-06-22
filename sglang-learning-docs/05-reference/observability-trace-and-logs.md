# Trace 与结构化日志联动查看请求生命周期

这篇记录本地查看 SGLang 请求 trace 的步骤，以及如何配合 `log_utils.py` 输出的结构化 JSON 日志观察 scheduler batch、`ReqToTokenPool` 和 `RadixCache` 状态。

## 1. 两类信息分别看什么

`trace` 适合看单个请求的生命周期：请求从 API server 进入、排队、prefill/extend、decode、返回结果，各阶段耗时和事件顺序可以在 Jaeger 里按同一个 trace 展开。

结构化日志适合看 scheduler 的周期性状态快照：某一刻 running batch 里有哪些 rid、forward mode 是什么、seq len / prefix len / req pool index 如何变化，`ReqToTokenPool`、KV pool 和 `RadixCache` 的容量与可驱逐 token 大概处于什么状态。

实际排查时通常先用 trace 找到“哪个请求在哪个阶段慢”，再用结构化日志按 rid 和时间点看当时 scheduler/cache 的状态。

## 2. 覆盖范围速查

| 问题 | 主要看哪里 | 能看到什么 |
|---|---|---|
| 请求是否进入系统、tokenize 是否慢 | Jaeger trace | `tokenize`、`api_server_dispatch`、request root span |
| 请求是否卡在队列 | Jaeger trace + scheduler JSON | `prefill_waiting` / `decode_waiting`，以及 `waiting_queue.size`、`queued_rids` |
| 当前 batch 是 prefill 还是 decode | scheduler JSON | `running_batch.forward_mode`、`prefix_lens`、`extend_lens`、`seq_lens` |
| 请求和 req pool 如何对应 | scheduler JSON | `running_batch.req_pool_indices`、每个 request 的 `req_pool_idx` |
| ReqToTokenPool 是否紧张、结构如何 | scheduler JSON | `size`、`alloc_size`、`available_size`、`used_size`、`free_slots_head/tail`、`active_rows` |
| KV pool 是否紧张 | scheduler JSON | `token_to_kv_pool_allocator.available_size`、`used_size`、`full_available_size`、`swa_available_size` |
| RadixCache 命中/占用的大致状态 | scheduler JSON | `radix_cache.total_size`、`evictable_size`、`protected_size`、`root_children` |
| MLX 后端执行是否慢 | Jaeger trace | `mlx_prefill`、`mlx_extend`、`mlx_decode`、`mlx_async_launch`、`mlx_async_finalize` |
| overlap/chained decode 是否发生 | Jaeger trace | `mlx.overlap.*` events、`mlx_chained_decode` |
| 请求参数、采样参数、输入输出内容 | request log | `--log-requests --log-requests-format json` |

这套 trace + scheduler JSON 覆盖的是“请求生命周期 + 核心调度和缓存状态”。它不会把每个 token 的完整 `ReqToTokenPool` 行内容、完整 Radix tree 结构、每层 KV tensor 内容都输出出来；这些信息通常体量太大，应该用断点、专门脚本或临时调试代码查看。

## 3. 启动 OTLP Collector 和 Jaeger

仓库已有 tracing compose：

```bash
docker compose -f examples/monitoring/tracing_compose.yaml up -d
```

Jaeger UI：

```text
http://127.0.0.1:16686
```

验证 collector 是否有服务：

```bash
curl -sS http://127.0.0.1:16686/api/services
```

结束后清理：

```bash
docker compose -f examples/monitoring/tracing_compose.yaml down
```

## 4. 启动带 trace 的服务

关键参数：

```text
--enable-trace
--otlp-traces-endpoint 127.0.0.1:4317
```

关键环境变量：

```text
SGLANG_TRACE_LEVEL=3
SGLANG_OTLP_EXPORTER_SCHEDULE_DELAY_MILLIS=500
SGLANG_OTLP_EXPORTER_MAX_EXPORT_BATCH_SIZE=64
```

MLX 后端示例：

```bash
PYTHONPATH=python \
SGLANG_USE_MLX=1 \
SGLANG_TRACE_LEVEL=3 \
SGLANG_OTLP_EXPORTER_SCHEDULE_DELAY_MILLIS=500 \
SGLANG_OTLP_EXPORTER_MAX_EXPORT_BATCH_SIZE=64 \
python/.venv/bin/python -m sglang.launch_server \
  --model-path mlx-community/Qwen3-0.6B-4bit \
  --disable-cuda-graph \
  --host 127.0.0.1 \
  --port 30000 \
  --enable-trace \
  --otlp-traces-endpoint 127.0.0.1:4317 \
  --skip-server-warmup
```

如果本机设置了 SOCKS 代理但 Python 环境缺 `socksio`，可以临时去掉 SOCKS 变量：

```bash
env -u ALL_PROXY -u all_proxy python/.venv/bin/python -m sglang.launch_server ...
```

触发请求：

```bash
curl -sS http://127.0.0.1:30000/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"trace hello","sampling_params":{"max_new_tokens":4,"temperature":0}}'
```

## 5. 在 Jaeger 里看 trace

UI 里选择 service `sglang`，点开最近的 trace。MLX 后端在 `SGLANG_TRACE_LEVEL=3` 下应该能看到这些 span 或 event：

```text
mlx_prefill
mlx_extend
mlx_decode
mlx_chained_decode
mlx_async_launch
mlx_async_finalize
mlx.overlap.finalize.begin
mlx.overlap.launch_fresh
mlx.overlap.launch_chained
mlx.overlap.chain_start
mlx.overlap.promote_chained
```

也可以直接查 Jaeger API：

```bash
curl -sS 'http://127.0.0.1:16686/api/traces?service=sglang&limit=20'
```

快速列出 MLX span/event：

```bash
curl -sS 'http://127.0.0.1:16686/api/traces?service=sglang&limit=20' \
  | python3 -c 'import json, sys
payload = json.load(sys.stdin)
spans, events = set(), set()
for trace in payload.get("data", []):
    for span in trace.get("spans", []):
        name = span.get("operationName", "")
        if name.startswith("mlx_"):
            spans.add(name)
        for log in span.get("logs", []):
            for field in log.get("fields", []):
                value = str(field.get("value", ""))
                if value.startswith("mlx."):
                    events.add(value)
print("spans:", sorted(spans))
print("events:", sorted(events))'
```

## 6. 打开 scheduler 状态 JSON 日志

`log_utils.py` 提供统一 JSON 输出能力，scheduler 状态由 `SchedulerStatusLogger` 周期性调用它写出。需要打开 metrics：

```bash
SGLANG_LOG_SCHEDULER_STATUS_TARGET=/tmp/sglang_scheduler_status \
SGLANG_LOG_SCHEDULER_STATUS_INTERVAL=1 \
python -m sglang.launch_server \
  ... \
  --enable-metrics
```

`SGLANG_LOG_SCHEDULER_STATUS_TARGET` 可以是目录，也可以是 `stdout`。目录模式会生成类似：

```text
/tmp/sglang_scheduler_status/<hostname>_<rank>.log
```

每条 `scheduler.status` JSON 现在包含：

```text
rank
running_rids
queued_rids
running_batch
waiting_queue
req_to_token_pool
token_to_kv_pool_allocator
radix_cache
```

状态日志会在 prefill/decode stats 路径触发，并受 `SGLANG_LOG_SCHEDULER_STATUS_INTERVAL` 节流。短请求排查时可以先把 interval 设小，例如 `0.2` 或 `0.5`。

其中 `running_batch` 里重点看：

```text
size
rids
forward_mode
forward_iter
seq_lens
req_pool_indices
prefix_lens
extend_lens
extend_num_tokens
out_cache_loc_len
requests
```

`req_to_token_pool` 里重点看：

```text
size
alloc_size
available_size
used_size
max_context_len
free_slots_len
free_slots_head
free_slots_tail
req_to_token_shape
active_rows
```

`active_rows` 是当前 running batch 的有界样本，不会输出完整二维表。每个元素对应一个 running request：

```text
rid
req_pool_idx
seq_len
token_locs_len
token_locs_head
token_locs_tail
```

这能帮助理解 `req_pool_idx` 如何定位到 `ReqToTokenPool.req_to_token` 的某一行，以及这一行里 token 位置如何指向 KV pool。`token_locs_head/tail` 只取前后少量元素，避免把大矩阵从 GPU 搬回 CPU。

`radix_cache` 里重点看：

```text
total_size
evictable_size
protected_size
full_evictable_size
swa_evictable_size
root_children
is_tree_cache
is_chunk_cache
supports_mamba
```

如果还需要请求体、sampling params、输入/输出内容，打开 request log：

```bash
python -m sglang.launch_server \
  ... \
  --log-requests \
  --log-requests-level 2 \
  --log-requests-format json \
  --log-requests-target stdout
```

`--log-requests-level 0` 只看 metadata，`1` 加 sampling parameters，`2` 加部分输入输出，`3` 打完整输入输出。线上环境慎用 level 3。

## 7. 联动排查技巧

先从 Jaeger 找到慢请求的 rid 和慢阶段。如果慢在 prefill/extend，看结构化日志里同一时间附近的 `waiting_queue.size`、`running_batch.forward_mode`、`prefix_lens`、`extend_lens` 和 `radix_cache.total_size`。如果慢在 decode，看 `running_batch.size`、`seq_lens`、`req_pool_indices` 和 KV pool 的 `available_size`。

`trace` 里的时间线是事件级别，适合回答“请求卡在哪里”；scheduler JSON 是状态级别，适合回答“当时 batch 和 cache 是什么样”。两者结合，比单独依赖普通文本日志更容易定位请求生命周期中的调度和缓存问题。

注意 `SGLANG_TRACE_LEVEL` 会过滤 span。MLX 相关 span 现在按 level 3 输出，默认排查建议使用 `SGLANG_TRACE_LEVEL=3`。

## 8. 收尾清理

本地验证结束后，停止并移除 OTLP collector、Jaeger 容器和 compose network：

```bash
docker compose -f examples/monitoring/tracing_compose.yaml down
```
