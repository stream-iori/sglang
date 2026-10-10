# 第三阶段：模型执行、采样与 Graph

> 源码基准：learning `2aa70e3eb4`，已合入 origin/main `6fc8d9da32`（2026-10-10）。

按当前源码走，每个阶段以可解释的证据验收，不要求每天固定耗时。

| 步骤 | 动作 | 验收 |
|---|---|---|
| 1 | TpModelWorker → ForwardBatch | 说明 input_ids / positions / seq_lens |
| 2 | Qwen3 层与 torch_native | 画出 Q/K/V 和 KV 读写 |
| 3 | ModelRunner._forward_raw | 列出 decode Graph / prefill Graph / eager 条件 |
| 4 | Sampler 与结果处理 | 解释新 ID 什么时候有 KV |
| 5 | RuntimeContext 与性能边界 | 解释 MPS 能验证什么、CUDA 还需什么 |

## 配套阅读

- [执行器](model-execution.md)
- [Graph](cuda-graph-and-padding.md)
- [配置分层](runtime-context.md)

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
