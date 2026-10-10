# Trace、指标和状态快照：三种证据一起看

> 源码基准：learning `2aa70e3eb4`，已合入 origin/main `6fc8d9da32`（2026-10-10）。

Trace 说明请求经历的阶段，状态日志说明当时的队列/批次/池，指标说明一段时间的分布与趋势。

```text
rid + 时间 + rank
  ├─ request trace：tokenize → waiting → forward → 输出
  ├─ scheduler.status：batch / queue / pool / cache
  └─ metrics：吞吐、排队、延迟等聚合统计
```

## 当前 RequestStage

| span | level | 意义 |
|---|---|---|
| tokenize | 1 | 编码阶段 |
| api_server_dispatch | 2 | API 分发 |
| prefill_waiting | 1 | 等待 prefill |
| prefill_forward | 1 | prefill 计算阶段 |
| chunked_prefill | 3 | 更细的 chunk 观察 |
| decode_forward | 1 | decode 计算阶段 |
| decode_loop | 3 | 更细的 decode 循环 |

stage 的确切起止还应看 req_time_stats 调用点，不能把 span duration 一律当作单个 kernel 时间。

## 启动观察

```bash
SGLANG_LOG_SCHEDULER_STATUS_TARGET=/tmp/sglang-status \
SGLANG_LOG_SCHEDULER_STATUS_INTERVAL=0.2 \
bash sglang-learning-docs/setup/launch_mac.sh --enable-metrics
```

TARGET 是输出目录，文件名由 hostname/rank 生成，也可用 `stdout`。Trace 需要可用的 OpenTelemetry 依赖与 OTLP collector。准备 collector 后可使用当前参数：

```bash
SGLANG_TRACE_LEVEL=3 bash sglang-learning-docs/setup/launch_mac.sh \
  --enable-trace --otlp-traces-endpoint 127.0.0.1:4317
```

本次没有运行 collector/Jaeger，因此这里是可读码的配置入口，不宣称 trace 端到端验证通过。标准 MPS 仍用普通请求阶段，不应编造另一套已测试的专属 span 名。

## 联动例子

| Trace 现象 | 联查状态 | 可能方向 |
|---|---|---|
| waiting 长 | queue、容量、子池预算 | admission、优先级、容量 |
| prefill 长 | extend token、命中、恢复 | 长输入、冷缓存、backend |
| decode 间隔大 | batch、通信、CPU 准备 | 提交、同步、负载 |
| PD transfer 长 | prealloc/transfer 队列 | 接收容量、连接、传输 |

这些是取证方向，不是仅凭现象就能成立的根因。状态日志有采样上限，多个 rank 的时间和 rid 也需要正确关联。

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/observability/req_time_stats.py](../../python/sglang/srt/observability/req_time_stats.py) | RequestStage 的名字与 level |
| [python/sglang/srt/observability/trace.py](../../python/sglang/srt/observability/trace.py) | OTLP/trace 实现 |
| [python/sglang/srt/utils/scheduler_status_logger.py](../../python/sglang/srt/utils/scheduler_status_logger.py) | 状态字段与采样 |
| [python/sglang/srt/arg_groups/fields/observability.py](../../python/sglang/srt/arg_groups/fields/observability.py) | 当前参数 |
