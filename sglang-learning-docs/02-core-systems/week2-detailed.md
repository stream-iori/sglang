# 第二阶段：批次、预算与统一缓存

> 源码基准：上游 `16eb87c06f`；本地集成 `4f0e1cbea2`（2026-10-20）。本地日志增强与验证范围见总索引和 Mac 验证记录。

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

## 五次练习的明确交付

| 次序 | 操作 | 留下的材料 | 验收问题 |
|---|---|---|---|
| 1 | 手绘双请求生命周期 | 每轮 Req/batch/plan 表 | 哪些对象持久，哪些只是本轮视图？ |
| 2 | 手算命中 A=4/6、B=0/3 | packed ID、position、长度 | 为什么 B 的 position 不从 2 开始？ |
| 3 | 真实 cold/warm | 输入、响应、cache 类型 | hit 在哪，剩余什么工作？ |
| 4 | 查 allocator/pool/Host 描述 | 地址和存储分层图 | 空槽、字节、buffer info 各是谁？ |
| 5 | 开启状态日志 | rid、请求行、cache 摘要 | 完成后 KV 不全释放是否正常？ |

答案入口分别是请求流、Prefill、统一缓存与状态日志章节。不要只抄字段；每个字段至少解释一次输入来源和下一步消费者。

扩展题：Full 足够但 SSM 槽不足时，为什么 admission 仍可失败？答案必须包含多个资源约束，不能只说“显存不够”。

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/managers/scheduler.py](../../python/sglang/srt/managers/scheduler.py) | 主循环 |
| [python/sglang/srt/model_executor/model_runner.py](../../python/sglang/srt/model_executor/model_runner.py) | 执行分支 |
