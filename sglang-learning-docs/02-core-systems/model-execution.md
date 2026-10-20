# ModelRunner 与 Runner：forward 现在怎样执行

> 源码基准：上游 `16eb87c06f`；本地集成 `4f0e1cbea2`（2026-10-20）。本地日志增强与验证范围见总索引和 Mac 验证记录。

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

## 当前 _forward_raw 的额外边界

```text
model.prepare_for_forward（模型可选 hook）
    → 模型 shared_experts 配置作用域 + ForwardContext
    → decode Graph 可用？是则执行并返回
    → live batch padding / 分布式元数据准备
    → deferred Mamba clear / COW（符合条件的 target extend）
    → SPLIT_PREFILL / prefill Graph / eager
```

这张图描述顺序约束。不要把本轮元数据准备、延后状态写入、模型计算都压缩成一个 `model.forward`，否则无法解释地址和 stream 依赖。

## 普通 Qwen3 层里实际算什么

| 阶段 | 张量含义 | 本轮为什么需要 |
|---|---|---|
| embedding | token ID → hidden vector | ID 本身不是带语义的浮点向量 |
| Q/K/V projection | 当前 token 的 query/key/value | query 读取历史，K/V 留给后续轮次 |
| RoPE / norm | 位置与归一化变换 | 模型定义的一部分，不能任意删除 |
| Attention | query 对有效历史 K/V 加权 | 请求边界和 mask 在这里起作用 |
| MLP / residual | 特征变换与残差 | 不是只做 Attention 就完成一层 |
| logits / sampling | 词表分数 → 输出 ID | temperature、top-p 等在输出选择路径生效 |

普通生成每次都执行所需的模型层；KV 缓存不会把后续 token 的整个模型计算省掉。

## 新的 RowBucketGraphRunner 放在哪里

当前 `RowBucketGraphRunner` 供 DeepSeek V4 replay tail 使用，对指定层范围按行数桶捕获 breakable Graph。它不是替换全部 eager/decode/prefill runner 的通用默认入口。

```text
模型专属 replay tail
    → 检查 warmup / 行数范围
    → 选 row bucket
    → 首次捕获或后续重放
    → 动态元数据读取留在 eager breaks
```

`ready()` 要等工作空间长到足够大小，避免捕获后更大的 batch 重新分配缓存，导致旧 Graph 仍引用已释放地址。理解这点比背“新增了一个类”更重要。

## 验收

用日志确认本地 model class、Attention、cache，再在 `_forward_raw` 指出 Mac 最终分支。答案应是当前固定配置下的普通 Torch 执行；不要因为 runner 目录包含 CUDA 类，就断言它们在本地参与了计算。

源码：[RowBucketGraphRunner](../../python/sglang/srt/model_executor/runner/row_bucket_graph_runner.py)、[模型调用者](../../python/sglang/srt/models/deepseek_v4_replay_tail.py)。

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/managers/tp_worker.py](../../python/sglang/srt/managers/tp_worker.py) | ForwardBatch 构造和执行 |
| [python/sglang/srt/model_executor/model_runner.py](../../python/sglang/srt/model_executor/model_runner.py) | _forward_raw 路由 |
| [python/sglang/srt/model_executor/runner/eager_runner.py](../../python/sglang/srt/model_executor/runner/eager_runner.py) | 普通执行 |
| [python/sglang/srt/layers/sampler.py](../../python/sglang/srt/layers/sampler.py) | 采样 |
| [python/sglang/srt/models/qwen3.py](../../python/sglang/srt/models/qwen3.py) | 本地模型结构 |
