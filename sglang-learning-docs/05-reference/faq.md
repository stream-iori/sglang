# 当前源码常见疑问

> 源码基准：上游 `16eb87c06f`；本地集成 `4f0e1cbea2`（2026-10-20）。本地日志增强与验证范围见总索引和 Mac 验证记录。

| 问题 | 回答 | 深入阅读 |
|---|---|---|
| ScheduleBatch 和 ForwardBatch 一样吗 | 前者偏调度和请求管理，后者面向本轮模型张量 | model-execution |
| get_next_batch_to_run 直接返回 batch 吗 | 当前返回 plan，从 batch_to_run 取执行批次 | request-batch-state-flow |
| 本地还是旧 RadixCache 吗 | Qwen3 MPS 实测是 Unified Radix Cache，从 registry 确认 | unified-cache-and-memory |
| 请求结束为什么 KV 不全释放 | 前缀缓存可继续占槽；请求行和 KV 生命周期不同 | scheduler-and-cache |
| Graph 是另一个模型吗 | 不是，是捕获/重放支持路径的执行方式 | cuda-graph-and-padding |
| prefill 一定 eager 吗 | 当前已有 prefill Graph 分支，取决于兼容性 | model-execution |
| eager 就没有 padding 吗 | backend/collective 对齐也可能需要 | cuda-graph-and-padding |
| token 是一个汉字吗 | 不一定，可完整字、多字、词片段或不完整字节片段 | tokenizer-internals |
| embedding 是每 byte 一个向量吗 | 按最终 token ID 查表；是否 byte 单位取决于分词 | tokenizer-internals |
| MPS 能验证 CUDA Graph 吗 | 不能，本地脚本普通 Torch MPS | Mac 指南 |
| 有 Rust 就替换 Python Scheduler 了吗 | Rust 模块职责不同，不能这样推断 | rust-serving-and-processing |
| 改 server_args 一定生效吗 | 解析发布后应检查有效配置 bag/override | runtime-context |
| KV 字节直接除 TP 吗 | 要检查分片和 KV head 复制规则 | multi-gpu |

## 本地首选读码链

```text
TokenizerManager → Scheduler normal → TpModelWorker
    → ForwardBatch → ModelRunner → EagerRunner
    → Qwen3 / torch_native → Sampler → DetokenizerManager
```

对应主题通过 [总索引](../README.md) 进入。版本与实际验证范围见 [Mac 验证](../setup/mac-validation.md)。

## 更新后新增疑问

| 问题 | 回答 |
|---|---|
| 最新是否意味着自动跟随 main？ | 本教程固定上游 16eb87c06f；之后更新要重新审计和验证 |
| 为什么还有本地与上游的差异？ | learning 保留 trace/日志增强，来源单独标注 |
| pool_host 现在存在吗？ | 存在；设备 pool/ 目录尚未建立，仍按实际文件阅读 |
| use_context 可以并发多线程跑模型吗？ | 不能据此推断；它切换进程全局引用，要求串行执行 |
| RowBucketGraphRunner 是所有模型的默认入口吗？ | 不是；当前调用来自 DeepSeek V4 replay tail |
| 并发 HTTP 通过能证明同一设备 batch 隔离吗？ | 还需日志证明同批；脚本只直接证明各自输出与串行参考一致 |
| check_docs 通过能证明全文语义吗？ | 不能；它检查结构、入口和指纹，语义仍需逐页审计 |

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/managers/scheduler.py](../../python/sglang/srt/managers/scheduler.py) | normal 主链 |
| [python/sglang/srt/mem_cache/registry.py](../../python/sglang/srt/mem_cache/registry.py) | cache 选择 |
| [python/sglang/srt/runtime_context.py](../../python/sglang/srt/runtime_context.py) | 有效配置 |
