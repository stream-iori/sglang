# 读 scheduler.status：把现象对到内存与批次

> 源码基准：上游 `16eb87c06f`；本地集成 `4f0e1cbea2`（2026-10-20）。本地日志增强与验证范围见总索引和 Mac 验证记录。

`scheduler.status` 是状态快照，不是逐条请求的完整事件历史。用同一 rank 的连续快照和请求响应一起判断。

## 启用

```bash
SGLANG_LOG_SCHEDULER_STATUS_TARGET=/tmp/sglang-scheduler-status \
SGLANG_LOG_SCHEDULER_STATUS_INTERVAL=0.2 \
bash sglang-learning-docs/setup/launch_mac.sh --enable-metrics
```

TARGET 可以是 `stdout` 或输出目录；目录模式下生成 `<hostname>_<rank>.log`。设置 TARGET 却不启用 metrics，源码会报错。不能把目录参数误当成指定 JSONL 文件名。

| 快照字段 | 应该看什么 |
|---|---|
| running_rids / queued_rids | 该快照批次中的请求与仍在排队的请求 |
| running_batch / cur_batch / last_batch | 大小、模式、长度；不是三份独立请求 |
| req_to_token_pool | 请求行使用情况与有效映射采样 |
| token_to_kv_pool_allocator | 空闲和容量计数；具体池可能有额外子预算 |
| radix_cache | 可回收/受保护等 cache 摘要 |

## 典型场景

| 现象 | 可能解释 | 必须补的证据 |
|---|---|---|
| queue 增长，batch 不增长 | 容量、策略、暂停、传输等待等 | admission / pool / engine 状态 |
| 请求完成，空闲 KV 未回初始值 | 缓存仍合法占用槽位 | 可驱逐缓存计数、cache 类型 |
| 同 prompt 第二次 extend 变短 | prefix 命中 | cached_tokens、输入 ID 与 cache key |
| 一个 rid 从 running 消失 | 完成、中止或回退 | finish reason / abort / retraction 日志 |
| batch size 周期变化 | 请求进入/退出、chunk、过滤 | 本轮模式、输出长度 |
| 映射槽位不连续 | allocator 的正常分配 | 请求行和 Attention 读表 |
| cur_batch 缺失 | 采样点/字段兼容性不同 | 当前 logger 实现和调度赋值点 |
| Full 空闲够但无法 admission | SWA/SSM 或共享预算紧 | 子池预算及 PrefillBudget |

本次修复 logger，读取 Scheduler 的 `cur_batch_for_debug`，对外仍输出 `cur_batch`。回归用例先复现该字段缺失，再验证修复；历史记录中的空值不能当作未执行的证明。

日志对 vector、请求和映射有采样上限，未显示的项不等于不存在。此处场景是诊断指南，不是捏造的线上快照。

## 读一条实际 JSON 的顺序

```text
event/rank/timestamp
    → running_batch.forward_mode / rids
    → 请求 input_len / output_len / extend_input_len
    → 请求行索引、有效映射 head/tail
    → allocator available / cache evictable、protected
    → 对照同次 HTTP response
```

`running_batch` 是传入 logger 的批次摘要，可能为 EXTEND 或 DECODE；不能仅凭字段名断定一定是持续 decode 集合。`cur_batch` 来自当前调度选择，`last_batch` 是上一轮记录，三者可能共享请求引用。

## 可执行观察

```bash
python/.venv/bin/python sglang-learning-docs/setup/verify_mac.py \
  --status-dir /tmp/sglang-scheduler-status --output /tmp/sglang-status-check
```

先用本页启动命令启动服务。验证脚本只选包含本次 run_id 的快照，避免把旧日志当作新证据；要求看到当前批次、请求行与 UnifiedRadixCache。

练习：如果脚本报告“没有状态事件”，先检查目录、metrics 开关和请求关联，再检查间隔。`0.2` 是低频观察；短请求可能被采样跳过。需要逐轮学习可临时把间隔设为 `0`，观测会增加开销，不用于性能比较。

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/utils/scheduler_status_logger.py](../../python/sglang/srt/utils/scheduler_status_logger.py) | 字段、采样限制和启用条件 |
| [python/sglang/srt/utils/log_utils.py](../../python/sglang/srt/utils/log_utils.py) | 日志目标 |
| [python/sglang/srt/managers/scheduler_components/metrics_reporter.py](../../python/sglang/srt/managers/scheduler_components/metrics_reporter.py) | 快照调用位置 |
| [python/sglang/srt/managers/scheduler.py](../../python/sglang/srt/managers/scheduler.py) | 当前 batch 赋值 |
