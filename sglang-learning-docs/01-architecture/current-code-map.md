# 从旧讲义到当前 SRT：哪些逻辑变了

> 源码基准：上游 `16eb87c06f`；本地集成 `4f0e1cbea2`（2026-10-20）。本地日志增强与验证范围见总索引和 Mac 验证记录。

**请求主线仍是编码、调度、模型执行、解码；变化集中在状态所有权、内存组织和执行分支。** 不用“文件很多”代替架构分析。

```text
messages / text
    → API + template + tokenizer
    → Req + NextBatchPlan + ScheduleBatch
    → TpModelWorker + ForwardBatch + KVLocPlan
    → ModelRunner → eager / decode graph / prefill graph
    → sampling → detokenizer → HTTP / SSE

配置：RuntimeContext        复用：UnifiedRadixCache
存储：allocator → pool      Host：buffer info → pool_host → storage
```

## 三个比较对象

| 对象 | 作用 | 能说明什么 |
|---|---|---|
| 早期 6 月讲义 | 历史知识来源 | 不能沿用旧路径、返回类型和启动结论 |
| 10 月 10 日集成 `2aa70e3eb4` | 本次更新前的本地源码 | 与之后未同步的本地 HEAD 相同，不代表上游没变化 |
| 本次集成 `4f0e1cbea2` | 当前教程依据 | 已合入固定上游 `16eb87c06f`，另有本地日志增强 |

上游 `6fc8d9da32..16eb87c06f` 的全仓库提交差为 677；SRT 路径有 687 个文件变化。数量只是审计范围，具体结论要读调用者、构造分支和状态更新。不要把全部变更都称为架构重写。

## 当前必须掌握的变化

| 层 | 当前机制 | 学习时必须回答 |
|---|---|---|
| 配置 | arg_groups 解析 → RuntimeContext bags | 输入值、有效值和模型私有值分别在哪里？ |
| 多模型串行嵌入 | create_context / use_context | 为什么可以切换上下文，却不能推断线程并发隔离？ |
| 调度 | NextBatchPlan、normal/overlap、延后结果与内存动作 | 当前 batch 与持续 decode 集合为何分开？ |
| 统一树 | TreeCore + Full/SWA/Mamba components | 结构匹配后，所需状态都可用吗？ |
| 地址 | allocation、KVLocPlan、KVIndexTranslator | 逻辑 ID 属于哪个 runner、哪个子池？ |
| Host 池 | DevicePoolInfo / buffer info → assembler → pool_host | 谁描述设备张量，谁决定容量和传输？ |
| 混合物理布局 | unified_memory_pool、unified_draft_pool、fused_draft | target 与 draft 的层、行和状态如何组合？ |
| 执行 | eager、decode/prefill Graph；模型专用 replay tail | 新执行器是通用默认，还是特定模型使用？ |
| Rust | server + processor + radix-tree | 前端处理与 Python 调度的边界在哪里？ |
| kernel | 按完整计算职责分组 | 新位置由算子职责决定，不能机械替换旧模型目录名 |

## 已迁移与设计方向要分开

`mem_cache/README.md` 中的分层图同时表达组织目标。当前 `allocator/`、`pool_host/` 已存在，设备池仍大量位于 `memory_pool.py` 等文件；本次检出没有 `pool/` 目录。读实际目录，再读设计图。

```text
设备提供者 → DevicePoolInfo + MHABufferInfo / MLABufferInfo / MambaStateBufferInfo
                                        ↓
            assembler：选层、容量、布局、传输输入
                                        ↓
                    pool_host/<family>.py → storage/<backend>
```

buffer info 是已有持久张量和布局的描述，不是新分配器，也不意味着布局都被压成相同格式。

## 本地增强与合并处理

| 内容 | 来源和处理 |
|---|---|
| struct_log、request trace helpers | learning 本地增强；保留，不当作上游标准功能介绍 |
| scheduler.status 扩展、metrics 调用 | learning 本地增强；补当前批次、请求池、allocator 和树摘要 |
| 当前批次字段 | 日志改读 Scheduler 的 cur_batch_for_debug，外部 JSON 仍叫 cur_batch |
| 历史 kernel / dflash 文件残留 | 核对无调用后删除；采用上游新目录及 DFlash V2 路径 |

## 读码练习与答案

1. 找到 normal 循环返回值，再找 `batch_to_run`。答案：选择结果是计划，计划中的 batch 才送入 run_batch。
2. 找到 `RowBucketGraphRunner` 的调用者。答案：当前调用来自 DeepSeek V4 replay tail，不能写成所有模型的必经节点。
3. 找到 Host 池构造中的 `buffer_info`。答案：设备描述和 Host 容量策略被拆开，Host 池不再必须抓住整个设备池对象。

```bash
rg -n 'batch_to_run|cur_batch_for_debug' python/sglang/srt/managers/scheduler.py
rg -n 'RowBucketGraphRunner' python/sglang/srt/models python/sglang/srt/model_executor
rg -n 'buffer_info=|device_capacity_tokens=' python/sglang/srt/mem_cache/hybrid_cache
```

继续阅读：[统一缓存](../02-core-systems/unified-cache-and-memory.md)、[执行器](../02-core-systems/model-execution.md)、[RuntimeContext](../02-core-systems/runtime-context.md)。
