# my-sglang

`my-sglang` 是一个 CPU Fake CUDA 的 SGLang 学习运行时。它不运行真实模型；它对齐基础 generation overlap 的调度顺序、`FutureMap`、FIFO result queue、request row、KV/page 与 radix cache 生命周期。

```text
CPU scheduler       Fake CUDA forward_stream            Fake CUDA copy_stream
launch B1     --->  B0 forward/sample -> FM[row]=t0
process B0    <---  B1 gather FM[row] -> sample t1       D2H B0 -> copy_done
```

## 核心语义

| 对象 | CPU Fake CUDA 表示 | 对齐的 SGLang CUDA 概念 |
|---|---|---|
| `MiniFutureMap.output_tokens_buf[row]` | `numpy.int64` 固定数组 | 设备侧 token relay buffer |
| `FakeCudaStream` | 延迟执行的 FIFO 任务队列 | `forward_stream` / `copy_stream` |
| `FakeCudaEvent` | `record()` / `synchronize()` 依赖门槛 | CUDA event |
| `result_queue` | 最大深度 2 的 FIFO | `batch.copy() + GenerationBatchResult` |

`pipeline_step()` 的不变量是：**先 enqueue 当前 B1，再等待并提交旧 B0。** 同一请求的 B1 通过 `FutureMap[row]` 读取 B0 的设备侧 token，不等 `B0` 写入 `Req.output_ids`。

## 范围

已覆盖：基础 prefill/decode、chunk、radix、retract、EOS/长度导致的多算 token 丢弃、延迟 row/KV 释放和失败 recovery。

不覆盖：CUDA 性能/线程并发、真实 kernel/PCIe、speculative、grammar、DP/TP、PP、分离式服务。Fake CUDA 验证的是控制流与数据依赖，不是物理性能。

## 测试与演示

```bash
uv run --extra test pytest -q

uv run my-sglang-generate \
  --input-ids 1,2 --token-ids 10,11,12,13 \
  --max-new-tokens 3 --overlap --trace
```

命令输出生成 token ids；`--trace` 输出 forward/copy stream 和 event 的实际模拟顺序。

推荐从 `tests/test_overlap_scheduler.py` 读取：它分别验证 FutureMap row buffer、event 的最小推进范围、`launch B1 -> process B0` 和失败恢复。

## 新人阅读入口

先读 [新人入门](docs/newcomer-guide.md)，再按 [代码阅读顺序](docs/code-reading-guide.md) 进入源码。需要查字段、KV page 和 radix cache 时看 [数据结构](docs/data-structures.md)；需要看 CPU/GPU 交替关系时看 [overlap 流水线](docs/overlap-pipeline.md)。
