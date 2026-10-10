# 第二阶段：批次、预算与统一缓存

> 源码基准：learning `2aa70e3eb4`，已合入 origin/main `6fc8d9da32`（2026-10-10）。

按当前源码走，每个阶段以可解释的证据验收，不要求每天固定耗时。

| 步骤 | 动作 | 验收 |
|---|---|---|
| 1 | Req / ScheduleBatch / ForwardBatch | 区分持久请求与本轮张量 |
| 2 | prepare_for_extend / allocation | 画出请求位置 → 槽位 |
| 3 | registry → UnifiedRadixCache | 确认本地实际类 |
| 4 | Full/SWA/Mamba / KVLocPlan | 列出锁、有效性、地址空间 |
| 5 | 重复 prompt 与状态日志 | 解释 cached_tokens 和可驱逐容量 |

## 配套阅读

- [状态链](request-batch-state-flow.md)
- [统一缓存](unified-cache-and-memory.md)

## 共用观察命令

```bash
rg -n "def event_loop_normal|def get_next_batch_to_run" python/sglang/srt/managers/scheduler.py
rg -n "can_run_graph|eager_runner.execute" python/sglang/srt/model_executor/model_runner.py
```

命令从仓库根目录执行。源码行号由当前搜索获得，避免复制旧版本固定行号。

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/managers/scheduler.py](../../python/sglang/srt/managers/scheduler.py) | 主循环 |
| [python/sglang/srt/model_executor/model_runner.py](../../python/sglang/srt/model_executor/model_runner.py) | 执行分支 |
