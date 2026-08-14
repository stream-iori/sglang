# my-sglang

`my-sglang` 是一个 CPU Fake CUDA 的 SGLang 学习运行时。它不运行真实模型；它对齐基础 generation overlap 的调度因果、`FutureMap`、FIFO result queue、request row、KV/page 与 radix cache 生命周期。它不是标准 SRT 的 API 或逐行复制。

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

当前核心路径不覆盖：CUDA 性能/线程并发、真实 kernel/PCIe、连续 prefill
overlap、分布式和分离式服务。Fake CUDA 验证的是控制流与数据依赖，不是物理性能。

## 测试与演示

```bash
uv run --extra test pytest -q

uv run my-sglang-generate \
  --input-ids 1,2 --token-ids 10,11,12,13 \
  --max-new-tokens 3 --overlap --trace
```

命令输出生成 token ids；`--trace` 输出 forward/copy stream 和 event 的实际模拟顺序。

推荐从 `tests/test_overlap_scheduler.py` 读取：它分别验证 FutureMap row buffer、event 的最小推进范围、`launch B1 -> process B0`、chunked prefill 交接和失败恢复。

## 新人阅读入口

先读 [新人入门](docs/newcomer-guide.md)，再按 [代码阅读顺序](docs/code-reading-guide.md) 进入源码。需要查字段、KV page 和 radix cache 时看 [数据结构](docs/data-structures.md)；需要看 CPU/GPU 交替关系时看 [overlap 流水线](docs/overlap-pipeline.md)；需要区分“概念对齐”与“实现简化”时看 [SRT 概念对照](docs/srt-concept-alignment.md)。标准 SRT 的进阶连续 prefill 流水线单独见 [prefill overlap](docs/prefill-overlap.md)。
