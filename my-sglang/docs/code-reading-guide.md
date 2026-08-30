# 文档优先，代码按需验证

先把文档读成一条故事，再把代码当作“这条箭头到底在哪儿发生”的证据。第一次阅读
不需要打开 `src/`；只有下表的问题仍答不上来时，才跳到最小的函数或测试。

## 推荐路线

1. [新人入门](newcomer-guide.md)：一条短请求的 token 因果。
2. [数据结构](data-structures.md)：请求、KV、page、cache 与两本 overlap 账。
3. [Scheduler 与 KV 概览](scheduler-kv-overview.md)：同步调度的一整个 round。
4. [模型执行连接层](model-execution-bridge.md)：把 `ForwardBatch`、Transformer、KV tensor 和 sampling 接成闭环。
5. [Transformer 基础概念](transformer-concept.md)：理解 runner 内部执行的模型数据流。
6. [动态流程](dynamic-flows.md)：把 chunk、cache、OOM、retract 放回生命周期。
7. [overlap pipeline](overlap-pipeline.md)：最后才看 CPU/Fake CUDA 错开一拍。
8. [SRT 概念对照](srt-concept-alignment.md) 与 [连续 prefill overlap](prefill-overlap.md)：理解教学边界和标准实现的额外复杂度。

## 看不懂时只查这一小段

| 仍然困惑的点 | 先回看的文档位置 | 最小代码验证 | 可执行证据 |
|---|---|---|---|
| 为什么最后一个 output 不在 KV？ | `data-structures.md` 的 KV 水位例子 | `MiniScheduleBatch.prepare_for_decode()` | `test_req_to_token_matrix_records_extend_and_decode_positions` |
| `ForwardBatch` 怎样变成模型计算？ | `model-execution-bridge.md` 的字段表 | `TinyTransformerModel.forward_batch()` | `test_forward_batch_flattens_multiple_requests_and_writes_given_slots` |
| K/V 为什么写到指定物理 slot？ | `model-execution-bridge.md` 的三层 KV 视角 | `TinyTransformerModel._forward_token()` | `test_decode_reads_history_and_overwrites_reused_slot` |
| radix 完整命中为何仍能采样？ | `model-execution-bridge.md` 的完整命中一节 | `_logits_from_cached_last_hidden()` | `test_full_prefix_hit_samples_from_cached_last_hidden` |
| 一轮为什么 prefill 优先？ | `scheduler-kv-overview.md` 的 round | `MiniScheduler.step()` | `test_prefill_priority_means_one_forward_batch_per_step` |
| chunk 为什么没有早期 output？ | `dynamic-flows.md` 的 chunk 时间线 | `_process_extend_result()` | `test_chunked_prefill_has_one_unfinished_request_and_no_early_output` |
| cache 命中为什么不重算 prefix？ | `data-structures.md` 的 radix 所有权 | `_attach_new_request()` | `test_radix_full_prompt_hit_allocates_no_extend_slots` |
| 为什么 B2 能先入队却不先执行？ | `overlap-pipeline.md` 的 event 一节 | `FakeCudaEvent.synchronize()` | `test_fake_cuda_event_only_advances_required_forward_prefix` |
| 为什么 finish 后不能立刻 free row/KV？ | `overlap-pipeline.md` 的延迟释放 | `_defer_finish()` | `test_pipeline_discards_speculative_extra_and_defers_release` |

想直接验证两条主线的闭环时，只运行：

```bash
uv run --extra test pytest -q tests/test_tiny_transformer.py
```

阅读代码时只追一条状态或一个 token；不要从类定义开始横向扫描所有字段。读完一个
函数后回到文档，能重新画出它改变的状态、KV 或队列，才继续下一处。
