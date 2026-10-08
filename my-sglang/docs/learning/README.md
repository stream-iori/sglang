# 从全局重新学 my-sglang

**字段看不懂时先看：[六张图读懂对象、字段与流程](15-field-atlas.md)。** F1 是对象总图，F2～F6 依次展开 Req、KV 地址、ForwardBatch、overlap 和回收；每图都有字段读写与标准 SRT 对照。

一次生成只有一个目标：**根据已有上下文，算出下一个 token；未结束就重复。** 调度器解决“谁先算、内存够不够”，模型解决“下一个 token 是什么”。

![一次生成的全局地图](assets/01-global-map.png)

图中 01～06 是后续章节的固定坐标。蓝色是请求与确认输出，绿色是 KV，紫色是模型，橙色是流水交接。先沿主箭头读一遍，再看下方三个优化分支。

## 先分清系统边界

```text
完整服务：文字请求 -> 网关选 Worker -> tokenizer -> Worker 内推理 -> token 转文字
                                               |
my-sglang 教学主线从 token ids 开始：              v
01 选请求 -> 02 准备 KV -> 03 打包输入 -> 04 模型 -> 05 确认输出
   ^                                                |
   +---------------- 未结束，继续 DECODE ------------+
                                                    |
                                                 已结束
                                                    v
                                             06 回收与缓存
```

| 概念 | 大白话 | 别混淆 |
|---|---|---|
| token id | 词表里一个符号的编号 | 编号 10 不代表数值大小或语义强弱 |
| Req | 一条请求跨多轮的记录 | 一个 batch 可以包含多条 Req |
| Scheduler | 决定本轮哪些请求可以算 | 不负责预测词 |
| KV cache | 已处理 token 的 K/V 向量，供后续 Attention 读取 | 不等于生成的 token 列表 |
| EXTEND / prefill | 给一段尚未处理的上下文计算 KV | retract 重建也走 EXTEND |
| DECODE | 每请求输入上次生成的一个 token，再预测一个 | 仍需读取历史 KV |
| forward / sampling | 先算词表分数，再选 token | 新 token 到下一轮才成为模型输入 |

## 按这个顺序学，不需要先背字段

| 顺序 | 页面 | 对应总图 | 读完能画什么 |
|---:|---|---|---|
| 1 | [一次生成与请求调度](01-request.md) | 01 → 05 → 01 | prompt 怎样接上逐 token 循环 |
| 2 | [KV、地址与 ForwardBatch](02-memory-batch.md) | 02 → 03 | 一个逻辑位置怎样找到物理 K/V |
| 3 | [模型怎样算下一个 token](03-model.md) | 04 | token → 向量 → logits → token |
| 4 | [分块、缓存与资源回收](04-branches.md) | 01、02、06 | 正常流程遇到长输入或内存压力怎么转弯 |
| 5 | [overlap 与失败恢复](05-overlap.md) | 03 → 04 → 05 的交接 | 执行侧先交 token，CPU 后确认 |

## 第二轮：把教学主线接到标准 SRT

读完前五篇后，按以下顺序进入真实服务。范围是基础自回归文本生成，不包含 speculative、DiT 等额外能力。

```text
第一轮总图                  第二轮放大图
03～04 工作单与模型  ------> S1 执行层 -> S2 Attention 后端
01、02、06 调度内存  ------> S3 持续组批 -> S4 mixed -> S5 两道预算门
04～05 输出与结束    ------> S6 采样到响应 -> S7 停止与增量输出
```

| 顺序 | 页面 | 需要补上的关系 |
|---:|---|---|
| 6 | [标准 SRT：真实执行链](06-standard-execution.md) | Scheduler、TpModelWorker、ModelRunner、Attention backend 如何协作 |
| 7 | [标准 SRT：持续组批与显存预算](07-standard-scheduling.md) | 谁加入 batch，KV 容量从哪里来，如何平衡本轮计算和长期内存 |
| 8 | [标准 SRT：从采样到流式响应](08-standard-output.md) | logits 怎样变成各请求的 token，再变成客户端能看到的文字 |

S1～S7 是前三层总图的局部放大，不是额外的七个运行阶段。图采用 ASCII，精确字段与数值可直接复制核对。

## 第三轮：从服务启动到设备执行与性能

```text
全局图之前： S8 进程关系 -> S9 启动与运行
节点 04：    S10 Graph -> S11 捕获档位与 padding
节点 04：    S12 TP rank -> S13 矩阵分片与通信
贯穿全程：   S14 客户端等待与设备时间线
```

| 顺序 | 页面 | 读完能解释什么 |
|---:|---|---|
| 9 | [启动与进程通信](09-startup-ipc.md) | 谁准备模型，谁传请求，rid 怎样把结果送回等待者 |
| 10 | [CUDA Graph 执行](10-cuda-graph.md) | capture/replay 复用什么，为什么需要缓冲区、padding 与兼容检查 |
| 11 | [基础 TP 与性能观察](11-tp-performance.md) | 一层计算怎样分片、为什么要通信、如何定位延迟与吞吐取舍 |

本轮沿用总图的职责边界，不引入 speculative、DiT 或其他模型扩展。性能部分是取证方法，不是对本机或线上运行状况的判断。

每页先看图，再手算一个例子，最后合上文档口述。答不出时回到对应箭头，不急着继续增加题量。

## 原文重新归位：按节点查，不再平铺阅读

第四轮先用以下贯穿案例检验前三轮知识，再按需回查原文。

| 顺序 | 页面 | 实际验证的关系 |
|---:|---|---|
| 12 | [一条请求的状态账本](12-request-ledger.md) | S15～S16：同步与 overlap 的输入、输出、KV 水位和结束时刻 |
| 13 | [三个请求共享有限 KV](13-shared-kv.md) | S17～S19：分块、前缀共享、retract 重建、逐页所有权与淘汰 |
| 14 | [从现象回到证据](14-evidence-lab.md) | S20：运行命令、检查记录、定位源码，再形成结论 |

```text
一条请求的正常推进 -> A/B 的共享与资源竞争 -> C 的缓存淘汰
           |                    |                    |
           +---------- 实际快照与源码核对 ------------+
```

配套可执行入口：[runtime_walkthrough.py](../../examples/runtime_walkthrough.py)。先自己预测下一轮，再运行查看结果。三个请求不要求同时运行，到达时刻已在案例中固定。

| 学习层 | 原有资料 | 什么时候打开 |
|---|---|---|
| 01 请求与调度 | [新人入门](../newcomer-guide.md)、[Scheduler 一轮](../scheduler-kv-overview.md) | 想跑 trace 或追容器交接 |
| 02～03 地址与工作单 | [数据结构](../data-structures.md)、[模型连接层](../model-execution-bridge.md) | 查字段、不变量和完整命中特例 |
| 04 模型 | [Transformer 概念](../transformer-concept.md)、[数学](../transformer-math.md) | 先解释各算子职责，再推公式 |
| 06 分支与回收 | [动态流程](../dynamic-flows.md) | 查 chunk、retract、cache 交接 |
| 流水交接 | [overlap](../overlap-pipeline.md) | 查 event、FIFO、延迟释放 |
| 真实实现边界 | [SRT 对照](../srt-concept-alignment.md)、[连续 prefill overlap](../prefill-overlap.md) | 主线掌握后再对照标准 SRT |
| 算子基础与实现 | [PyTorch](../pytorch-concept.md)、[Triton 概念](../triton-concept.md)、[CUDA 映射](../triton-cuda-basics.md)、[Transformer kernels](../triton-transformer.md)、[示例](../../examples/triton/README.md) | 从节点 04 往下钻 shape、stride、GPU 执行 |
| 多 Worker 服务 | [SMG 边界](../smg-capability-map.md)、[本地请求链](../local-chat-completions-flow.md)、[学习路线](../smg-learning-path.md)、[Fake Worker 实验](../smg-fake-worker-lab.md) | 从节点 01 往外扩展路由与服务 |
| Rust 按需补课 | [所有权](../rust-lang/ownership-move-and-borrowing.md)、[模块](../rust-lang/modules-visibility-and-crate.md)、[workspace](../rust-lang/workspace.md)、[derive](../rust-lang/common-derive.md)、[Arc/Weak](../rust-lang/arc-weak-and-cycle-references.md)、[借用切片](../rust-lang/borrowed-slices-and-vec-type-annotation.md)、[模式匹配](../rust-lang/if-let-ref-and-deref.md) | 阅读 SMG 时按语法卡点查询 |
| 验证与复习 | [代码索引](../code-reading-guide.md)、[复习题](../../revise/runtime-review.md) | 用代码或题目验证某一条具体关系 |

## 教学范围

脚本 runner 用预设 token 解释控制流；Fake CUDA 用 CPU 队列模拟异步依赖；tiny runner 用 NumPy 真正计算单层 Transformer。tiny 当前不与 overlap 组合。这里的图不表示 GPU 性能或真实服务部署拓扑。

图像生成方式、原始提示词与尺寸见 [配图记录](assets/prompts.md)。
