# ModelRunner 与 Runner：forward 现在怎样执行

> 源码基准：learning `2aa70e3eb4`，已合入 origin/main `6fc8d9da32`（2026-10-10）。

`forward` 是每轮模型执行的入口，内部按模式、设备和兼容性选择执行器。Graph 是一种执行方式，也在 forward 路径中。

```text
ScheduleBatch
   ↓ TpModelWorker.forward_batch_generation
ForwardBatch.init_new
   ↓ ModelRunner.forward / _forward_raw
   ├─ decode graph 可执行？ → DecodeCudaGraphRunner.execute
   └─ 准备 live batch：分布式 padding、元数据、必要的状态操作
        ├─ SPLIT_PREFILL → 分层执行路径
        ├─ prefill graph 可执行？ → PrefillCudaGraphRunner.execute
        └─ EagerRunner.execute
   ↓ logits 处理 / 采样
GenerationBatchResult → Scheduler
```

## 本轮张量从哪来

| 数据 | 用途 |
|---|---|
| input_ids | 查 embedding；不是 CPU 每轮重新发送整段文字 |
| positions | RoPE 等位置计算 |
| seq_lens / extend_seq_lens | 历史长度、本轮 query 边界 |
| req_pool_indices | 选择请求的 KV 映射行 |
| out_cache_loc / KVLocPlan | 本轮 KV 写入及读地址 |
| sampling_info | temperature、top-p、停止与约束相关信息 |

## 模型内部

```text
token ID → embedding
   → [norm → Q/K/V + RoPE → Attention → 残差 → MLP → 残差] × L
   → final norm → lm_head → logits → sampler → 下一个 ID
```

普通生成每轮执行多层 Transformer。Graph 重放也是执行这些层的算子，不是“每重放一次只算下一层”。KV 缓存省去历史位置 K/V 的重复计算，Graph 主要减少可捕获路径的提交开销，两者作用不同。

## 标准 Torch MPS

| 项目 | 本地配置 |
|---|---|
| worker / runner | 标准 TpModelWorker / Torch ModelRunner |
| Attention | torch_native |
| sampling | pytorch |
| 调度 | normal，关闭 overlap |
| CUDA Graph | 不执行 |

`torch_native_backend.py` 可以直观看到逐请求的 Attention 切片。它便于学习和设备兼容，不应据此推断 CUDA 优化 backend 也逐请求 Python 循环。

详细机制见 [Graph 和 padding](cuda-graph-and-padding.md)。

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/managers/tp_worker.py](../../python/sglang/srt/managers/tp_worker.py) | ForwardBatch 构造和执行 |
| [python/sglang/srt/model_executor/model_runner.py](../../python/sglang/srt/model_executor/model_runner.py) | _forward_raw 路由 |
| [python/sglang/srt/model_executor/runner/eager_runner.py](../../python/sglang/srt/model_executor/runner/eager_runner.py) | 普通执行 |
| [python/sglang/srt/layers/sampler.py](../../python/sglang/srt/layers/sampler.py) | 采样 |
| [python/sglang/srt/models/qwen3.py](../../python/sglang/srt/models/qwen3.py) | 本地模型结构 |
