# my-sglang

`my-sglang` 是一个 CPU Fake CUDA 的 SGLang 学习运行时。它用确定性脚本解释 generation
overlap，也可以运行一个固定权重的 NumPy 单层 Transformer，把 `ForwardBatch`、物理 K/V、
logits 和 sampling 接成教学闭环。它对齐基础调度因果、`FutureMap`、FIFO result queue、
request row、KV/page 与 radix cache 生命周期，但不是标准 SRT 的 API 或逐行复制。

```text
稳定 decode turn（进入时 B0 已在队列）

CPU scheduler             Fake CUDA 的必要 stream 前缀
enqueue B1                forward queue: [... B0, B1]
wait B0.copy_done   --->  B0 forward/sample -> FM[row]=t0 -> D2H B0
process B0                B1 仍在队列，没有被 B0 event 提前执行

后续 wait B1.copy_done
                    --->  B1 gather FM[row]=t0 -> sample t1 -> D2H B1
```

## 核心语义

| 对象 | CPU Fake CUDA 表示 | 对齐的 SGLang CUDA 概念 |
|---|---|---|
| `FutureMap.output_tokens_buf[row]` | `numpy.int64` 固定数组 | 设备侧 token relay buffer |
| `FakeCudaStream` | 延迟执行的 FIFO 任务队列 | `forward_stream` / `copy_stream` |
| `FakeCudaEvent` | `record()` / `synchronize()` 依赖门槛 | CUDA event |
| `result_queue` | 最大深度 2 的 FIFO | `batch.copy() + GenerationBatchResult` |

`pipeline_step()` 的不变量是：**先 enqueue 当前 B1，再等待并提交旧 B0。**
B1 执行时通过 `FutureMap[row]` 读取 B0 的设备侧 token，不等 `B0`
写入 `Req.output_ids`。Fake CUDA 的 event 只推进必要前缀；真实 CUDA
则可能在 CPU 等待前已自主执行部分已提交工作。

## 范围

已覆盖：基础 prefill/decode、chunk、radix、retract、EOS/长度导致的多算 token 丢弃、延迟 row/KV 释放和失败 recovery。

当前核心路径不覆盖：真实模型权重和 tokenizer、CUDA 性能/线程并发、真实 kernel/PCIe、
连续 prefill overlap、分布式和分离式服务。Fake CUDA 验证的是控制流与数据依赖；NumPy
教学模型验证的是 shape、KV 映射和 token 因果，两者都不代表物理性能或模型质量。

## 测试与演示

```bash
uv run --extra test pytest -q

uv run my-sglang-generate \
  --input-ids 1,2 --token-ids 10,11,12,13 \
  --max-new-tokens 3 --overlap --trace
```

命令输出生成 token ids；`--trace` 输出 forward/copy stream 和 event 的实际模拟顺序。

要让同一个同步 Scheduler 真正执行一个 NumPy 单层 Transformer，而不是读取预设 token：

```bash
uv run my-sglang-generate \
  --input-ids 1,2 --runner tiny --max-new-tokens 3 --trace
```

它会真实执行 embedding、Attention、SwiGLU、物理 KV slot 读写、LM head 和 greedy sampling；
固定权重下输出 `94,94,94`。首版教学模型不与 `--overlap` 组合。

推荐从 `tests/test_overlap_scheduler.py` 读取：它分别验证 FutureMap row buffer、event 的最小推进范围、`launch B1 -> process B0`、chunked prefill 交接和失败恢复。

## 文档优先学习入口

这套项目可以先不读代码。主线一解释运行时怎样组织请求和 KV，连接层把 `ForwardBatch`
送进模型，主线二解释 Transformer 怎样算出下一个 token；PyTorch、Triton 和 CUDA 是按需
查阅的基础或进阶材料。

| 顺序 | 文档 | 读完应能说清楚什么 |
|---:|---|---|
| 1 | [新人入门](docs/newcomer-guide.md) | 一次生成为何是 `prompt -> 首 token -> decode`，以及 token 去了哪里。 |
| 2 | [数据结构、所有权与不变量](docs/data-structures.md) | `Req`、row、KV slot/page、cache 和 `FutureMap` 分别保存什么。 |
| 3 | [Scheduler 与 KV 概览](docs/scheduler-kv-overview.md) | 一轮调度怎样先选请求、再分配、执行、提交和交接。 |
| 4 | [动态流程](docs/dynamic-flows.md) | chunk、cache 命中、KV 压力、retract、失败恢复时哪些状态保留。 |
| 5 | [模型执行连接层](docs/model-execution-bridge.md) | `ForwardBatch` 如何驱动 Transformer、物理 KV 读写、logits 和 sampling。 |
| 6 | [Transformer 基础概念](docs/transformer-concept.md) | 用直观语言理解 hidden state、decoder-only block、FFN、Attention 与 KV cache。 |
| 7 | [Transformer 核心数学](docs/transformer-math.md) | 沿数据流读懂 RMSNorm、Q/K/V、RoPE、Attention、SwiGLU、LM head 和 sampling。 |
| 8 | [Fake CUDA overlap pipeline](docs/overlap-pipeline.md) | 为什么能先 launch 当前 batch、再 FIFO 处理上一 batch。 |
| 9 | [与标准 SGLang SRT 的概念对照](docs/srt-concept-alignment.md) | 教学实现对齐了什么、刻意没有实现什么。 |
| 10 | [进阶：标准 SRT 的连续 prefill overlap](docs/prefill-overlap.md) | 为什么连续 prefill overlap 需要额外的在途 chunk 账本。 |
| 11 | [PyTorch 基础概念](docs/pytorch-concept.md) | Tensor、shape/dim、stride/contiguous、广播、归约、dtype 与 device。 |
| 12 | [Triton 与 CUDA：真实 GPU kernel](docs/triton-cuda-basics.md) | CUDA 执行模型与 Triton program 的边界、映射和真实运行方式。 |
| 13 | [Triton 示例中的 Transformer 数据流](docs/triton-transformer.md) | 按 block 执行顺序理解 02～07 示例覆盖什么、还缺什么。 |
| 14 | [Triton 基础概念](docs/triton-concept.md) | contiguous、逻辑顺序、底层存储与一维 pointer offset 的关系。 |

[文档优先、代码按需验证的索引](docs/code-reading-guide.md) 把每一处“卡住时该看哪里”
压缩成最小跳转路径。代码和测试不是前置阅读任务，而是用来验证某个具体箭头或不变量。
