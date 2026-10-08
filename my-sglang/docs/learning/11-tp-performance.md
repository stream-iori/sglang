# 标准 SRT（六）：基础 TP 多卡与性能观察

[返回全局地图](README.md) · 上一页：[CUDA Graph](10-cuda-graph.md)

下一页：[贯穿实验：一条请求的状态账本](12-request-ledger.md)。

**TP 让多张 GPU 合作计算同一个模型调用；收益取决于计算减少量与通信成本。** 本篇放大总图 04，限定普通 dense Transformer 的基础 TP，暂不深入 PP、DP、MoE 或其他并行方式。源码基准为 `a5a7123f54`。

## 图 S12：一个服务的多个 TP rank

```text
同一组请求 / 一次模型调用
            |
       协调请求与执行
            |
       +----+----------------+
       v                     v
   rank 0 / GPU 0         rank 1 / GPU 1
   权重的一部分           权重的另一部分
       |                     |
       +---- 必要通信合并 -----+
                   |
              继续下一层
```

TP rank 不是两个互不相关的模型服务。相关 rank 必须对协作计算和通信形成一致安排，不能各自随意选择不同请求继续执行。

普通路径由相应入口 rank 接请求，再向参与 rank 广播；配置不同会改变通信组。证据：[RequestReceiver](../../../python/sglang/srt/managers/scheduler_components/request_receiver.py)、[Scheduler 进程启动](../../../python/sglang/srt/entrypoints/engine.py)。

## 图 S13：把矩阵乘拆成两种基本方式

下面采用源码注释的数学记法 `Y = XW`；不要直接将其列/行名称套到 PyTorch 权重底层存储维度。

```text
按输出维拆 W：W=[W0 | W1]
   GPU0：Y0=XW0      GPU1：Y1=XW1
                 |
   需要完整输出时拼成 Y=[Y0 | Y1]（all-gather）
   下一层能消费分片时，可以暂不拼接

按输入维拆 W、X：W=[W0; W1]，X=[X0 | X1]
   GPU0：Z0=X0W0    GPU1：Z1=X1W1
                 |
          Y=Z0+Z1（求和归约，典型为 all-reduce）
```

| 通信 | 大白话 | 对应上图 |
|---|---|---|
| all-gather | 把各卡不同片段收齐 | 合成完整输出特征 |
| all-reduce | 合并各卡对同一结果的部分贡献，并让参与者得到结果 | 将局部乘积相加 |

真实路径可推迟、融合或采用其他等价通信安排，不是每个线性层都无条件调用一次 all-gather 加一次 all-reduce。证据：[`ColumnParallelLinear`](../../../python/sglang/srt/layers/linear.py)、同文件 `RowParallelLinear`。

## KV 不一定随 TP 数严格等比分摊

| 普通示例 | 每 GPU 的 KV heads |
|---|---:|
| 总 KV heads=8，TP=2 | 4 |
| 总 KV heads=8，TP=8 | 1 |
| 总 KV heads=2，TP=4 | 不能理解成 0.5 个；此类布局涉及复制 |

当前 `get_num_kv_heads()` 在基础 TP 情形使用至少 1 个 head 的下界。因此 KV 字节数应按实际每设备 heads 计算；不能把单卡 KV 除以 TP 当作所有模型的固定规律。模型、后端与并行布局也有兼容约束。

证据：[`ModelConfig.get_num_kv_heads()`](../../../python/sglang/srt/configs/model_config.py)。单 token KV 的手算继续用[显存预算公式](07-standard-scheduling.md)。

## 图 S14：把性能放回请求时间线

```text
客户端发送 -> 输入处理 / 排队 -> prefill -> 传回首片段 -> 后续片段 ... -> 结束
|<--------------- 客户端首输出等待 ---------------->|
                                                     |<--片段间隔-->|

设备某轮：输入准备 -> 本地算子 / 历史KV读取 -> 必要通信 -> 输出交接
```

客户端片段不一定等于一个 token；服务内部指标的起止点也可能不同。做比较前先写明测量位置与统计单位。

| 想判断什么 | 最少需要的证据 | 不足以单独说明问题的现象 |
|---|---|---|
| 首输出慢在哪里 | 请求时间点、排队与 prefill 记录、输出链时间点 | 只有客户端总耗时 |
| TP 通信是否抵消收益 | 相同负载下各 rank 的计算与通信时间线 | “用了两张卡” |
| KV 是否成为容量压力 | 池占用、活跃/缓存页、retract 与准入情况 | 总显存使用率高 |
| Graph 是否有帮助 | 相同请求条件下执行路径和耗时对照 | Graph 初始化成功日志 |
| 吞吐改善是否伴随等待变差 | 吞吐及首输出/后续输出延迟分布一起看 | 单一平均 tokens/s |

标准源码的 TTFT 与输出间隔指标入口：[TokenizerManager](../../../python/sglang/srt/managers/tokenizer_manager.py)、[MetricsCollector](../../../python/sglang/srt/observability/metrics_collector.py)。

对比实验至少保持模型、输入输出长度、并发/到达方式、cache 冷热与采样条件可比；报告完成请求数和延迟分布，避免只挑最快一次。这里给出观察方法，不替代真实 GPU 性能实验。

## 合上文档自检

| 问题 | 答案要点 |
|---|---|
| TP=2 是否代表两条独立请求各在一张卡运行？ | 不代表；TP 的核心是合作执行同一模型计算。 |
| 两卡理论算力翻倍，单请求延迟一定减半吗？ | 不一定，有通信、内存访问和其他固定开销。 |
| KV heads 少于 TP 数时，还能机械除以 TP 吗？ | 不能，需考虑每卡至少一个 head 等实际分片/复制规则。 |

至此可以把启动、进程消息、单轮执行、Graph、多卡通信与请求可见延迟放回同一张全局图。
