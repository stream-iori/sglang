# 标准 SRT（三）：从采样到流式响应

[返回全局地图](README.md) · 上一页：[持续组批与显存](07-standard-scheduling.md)

下一页：[启动与进程通信](09-startup-ipc.md)。

**模型得到分数、CPU 确认 token、客户端看到文字，是三个不同时间点。** 本篇接上总图 04 → 05 → 06 以及客户端返回链，采用普通 Python 服务路径。

## 图 S6：同一条请求，从文字进去再变回文字

```text
客户端 messages
    |
服务入口：校验、chat template 等准备
    |
TokenizerManager：tokenize / 校验输入，按 rid 管理等待状态
    |
Scheduler -> 模型 forward -> logits -> 采样 token
    ^                                    |
    |                          异步路径等待结果可读
    |                                    |
    +--未停止，准备后续计算-- CPU 更新 Req / 判断停止
                                         |
                                OutputStreamer 选择可发送增量
                                         |
                                DetokenizerManager 增量解码
                                         |
                                TokenizerManager 按 rid 分发
                                         |
                                 API 返回 JSON / SSE
```

这是职责图；同步和 overlap 的精确先后按[交接章节](05-overlap.md)理解。不是每个框都新建一个进程，也不是每生成一个 token 必须立即发一条 SSE。

更完整的服务与进程关系见[本地请求全链路](../local-chat-completions-flow.md)。

## 同一批请求，可以各自采用不同采样参数

| 请求 | 参数示例 | 含义 |
|---|---|---|
| A | greedy | 选最高分 token |
| B | temperature=0.7，top_p=0.9 | 调整分布，再从截取的候选中采样 |
| C | top_k=20 | 限制候选数量，再按相应分布采样 |

```text
各请求的 logits 行
       |
按各请求配置处理惩罚、采样参数等
       |
greedy 或概率采样
       |
每请求一个 next token -> 跟随相同的请求顺序返回
```

| 参数/对象 | 大白话 |
|---|---|
| temperature | 改变分布尖锐程度，普通概率路径中缩放 logits |
| top-k | 限制排名靠前的候选数量 |
| top-p | 按概率排序，保留累计概率达到阈值的候选集合 |
| repetition / frequency / presence penalty | 根据历史出现情况调整候选偏好，具体规则不同 |
| SamplingBatchInfo | 把每请求参数、必要的历史状态组织成批处理数据 |

上图是概念顺序，不要求所有 backend 都物化每个中间张量。batch 过滤/合并时，采样参数与状态也要跟着变，不能把 A 的参数错配给 B。当前 Sampler 的全 greedy 批有专门 argmax 路径。

证据：[`SamplingBatchInfo`](../../../python/sglang/srt/sampling/sampling_batch_info.py)、[`Sampler`](../../../python/sglang/srt/layers/sampler.py)。公式继续查[Transformer 数学](../transformer-math.md)。

## token 已确认，为什么还不能直接当一段文字发送

```text
生成 token 编号 -> 更新可见 token 范围 -> 解码所需上下文与偏移
                                              |
                                   得到新形成的文字片段
                                              |
                                   处理停止内容 -> 对外返回
```

一个 token 可能只是字符的部分编码或一个词的片段。不能假定 `decode([a]) + decode([b])` 总等于 `decode([a,b])`。增量解码维护已读位置和周边上下文，避免重复输出或把尚未完整的内容提前输出。

`output_ids` 是内部确认的 token 历史；最终可见内容还受停止位置、特殊 token 处理和流式策略影响。不要把内部列表长度直接当文字长度。

证据：[`DetokenizerManager`](../../../python/sglang/srt/managers/detokenizer_manager.py)、[`Req.init_incremental_detokenize()`](../../../python/sglang/srt/managers/schedule_batch.py)。

## 图 S7：停止条件有 token 层，也有文字层

| 条件 | 检查对象 | 例子 |
|---|---|---|
| 最大生成长度 | 已生成数量 | 达到 max_new_tokens |
| EOS / stop token | token 编号 | 命中停止编号；具体受 ignore_eos 等配置影响 |
| stop string | 解码出的文本 | 生成文本包含 `END`，可能跨多个 token |
| abort | 请求生命周期 | 客户端断开或显式取消 |

```text
假设停止字符串是 END
解码尾部只有 E  -> 可能是停止词前缀，暂缓发送相关输出
后续拼成 END   -> 判断停止，按配置裁剪停止内容
后续变成 Else  -> 不匹配 END，继续正常输出
```

例子描述文字关系，不指定真实 tokenizer 如何切分。当前 OutputStreamer 会结合 stream interval 和 stop-string prefix 判断是否发送；Detokenizer 有停止内容裁剪逻辑。因此“CPU 已确认”不等于“客户端已收到”。

证据：[`Req.check_match_stop_str_prefix()` / `_check_str_based_finish()`](../../../python/sglang/srt/managers/schedule_batch.py)、[`OutputStreamer`](../../../python/sglang/srt/managers/scheduler_components/output_streamer.py)、[`trim_matched_stop()`](../../../python/sglang/srt/managers/detokenizer_manager.py)。

## 取消、正常结束、故障恢复，不能共用一个假设

```text
客户端断开 -> TokenizerManager 检测 / abort task
                                |
                           发送 abort 请求
                                |
                      Scheduler 处理请求终止与资源生命周期
```

| 场景 | 正确理解 |
|---|---|
| 排队中取消 | 不应继续为该请求做正常准入和生成 |
| 执行中取消 | 要经过服务的取消与资源处理流程，不等于即时撤销已经提交的 GPU kernel |
| 已向客户端发送部分文字 | 无法撤回已经发送的内容；停止继续生成与发送 |
| 教学 Fake CUDA 异常 | 教学代码可 discard 在途结果，保留确认输出并重排 |
| 标准设备/进程异常 | 不能套用教学 recovery；是否可恢复要看具体异常处理路径 |

标准 `event_loop_overlap()` 没有与教学 `_recover_pipeline_failure()` 等价的通用捕获并重新排队逻辑。这里只给出边界，不承诺任意 CUDA 异常都能恢复。

证据：[`TokenizerManager.abort_request()` / `create_abort_task()`](../../../python/sglang/srt/managers/tokenizer_manager.py)、[`Scheduler.event_loop_overlap()`](../../../python/sglang/srt/managers/scheduler.py)。

## 合上文档自检

| 问题 | 答案要点 |
|---|---|
| batch 合并只拼 input_ids 够吗？ | 不够，row、长度、采样参数和状态必须与请求顺序一致。 |
| 新 token 已写入 output_ids，用户一定看到了吗？ | 不一定，还要经过发送策略、增量解码、API 和网络。 |
| stop string 与 stop token 是同一种匹配吗？ | 不是，前者匹配文字，可能跨 token；后者匹配编号。 |

读完本轮，可以串起：**谁来算 → 地址在哪 → 后端怎样算 → 怎样选 token → 哪些文字能发送 → 何时结束回收。**
