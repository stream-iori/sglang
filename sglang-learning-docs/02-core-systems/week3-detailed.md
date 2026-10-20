# 第三阶段：模型执行、采样与 Graph

> 源码基准：上游 `16eb87c06f`；本地集成 `4f0e1cbea2`（2026-10-20）。本地日志增强与验证范围见总索引和 Mac 验证记录。

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

## 五次练习的明确交付

| 次序 | 操作 | 留下的材料 | 验收问题 |
|---|---|---|---|
| 1 | worker → ForwardBatch | input_ids/positions/seq_lens 表 | 哪些位置本轮新算？ |
| 2 | Qwen3 层与 native Attention | Q/K/V 与 pool 读写图 | 哪些历史计算被 KV 省掉？ |
| 3 | _forward_raw 路由 | 分支图，标 Mac 实际分支 | Graph 是新模型还是执行方式？ |
| 4 | 采样 → 下一轮输入 | y0/y1 的时序表 | y0 生成时，自己的 KV 是否已有？ |
| 5 | 配置和资源作用域 | 原始值、有效值、模型私有值 | use_context 为什么要求串行？ |

Graph 专题先做读码推演：固定桶 8、有效请求 5，标出三个 padding 位置怎样被忽略。需要 GPU 的实际 capture/profile 留到对应硬件阶段，不能用 MPS 运行时长代替。

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/managers/scheduler.py](../../python/sglang/srt/managers/scheduler.py) | 主循环 |
| [python/sglang/srt/model_executor/model_runner.py](../../python/sglang/srt/model_executor/model_runner.py) | 执行分支 |
