# 动手练习：玩具模型帮助理解，真实服务负责验证

> 源码基准：上游 `16eb87c06f`；本地集成 `4f0e1cbea2`（2026-10-20）。本地日志增强与验证范围见总索引和 Mac 验证记录。

三个 demo 是缩小模型，不是当前 SRT 的复制品。它们只解释队列、前缀和消息流；真实地址管理以统一缓存与 KVLocPlan 为准。

| Demo | 运行命令 | 能学到 | 没有实现 |
|---|---|---|---|
| scheduler | python/.venv/bin/python sglang-learning-docs/06_demo_scheduler.py | 请求入队、extend/decode、结束 | GPU、真实预算、统一 cache、Graph |
| prefix tree | python/.venv/bin/python sglang-learning-docs/06_demo_radix_cache.py | 相同前缀与分叉 | 组件锁、物理池、驱逐、Host |
| ZMQ | python/.venv/bin/python sglang-learning-docs/06_demo_zmq_pipeline.py | 消息经过分词/执行/解码 | 真实 tokenizer、模型、HTTP |

ZMQ demo 用 ord/chr 作玩具编码，不等于 byte-level BPE；它可能自动切到 inproc 线程 fallback，输出中应确认实际路径。

## 四个练习

| 编号 | 任务 | 验收 |
|---|---|---|
| 1 | 画两个请求加入/结束的批次变化 | 说明 Req 持续存在但 batch 改变 |
| 2 | prefix tree 输入 [1,2,0] | 解释匹配 [1,2]，剩余 [0] |
| 3 | 把玩具对象映射到当前源码 | 指出 ScheduleBatch、ForwardBatch、plan 的差别 |
| 4 | 跑真实 verify_mac.py | 保存 cold/warm、chat、SSE 结果 |

## 扩展

用模型 tokenizer 打印中文、emoji 的字节与 token；把请求中的最大输出长度改小；对照输出 ID 数与终止原因。不要通过随机改核心资源释放逻辑来“验证理解”。

参考答案见 [exercise-solutions](exercise-solutions.md)。

## 当前版本的结业前实验

| 编号 | 实验 | 交付与验收 |
|---|---|---|
| 5 | A=6 token、命中 4；B=3、命中 0 | packed 5 个 query，positions 分别 4..5 与 0..2 |
| 6 | 两个不同输入串行/并发生成 | 各自 greedy ID 一致，后续请求仍可用；是否同批由日志判断 |
| 7 | 启用 status 后运行 verify --status-dir | 同 run_id、有效 cur_batch、请求行、实际 cache |
| 8 | 区分三个配置动作 | publish、override、use_context 的作用与恢复范围 |
| 9 | 画新 Host 构造链 | provider → buffer info → assembler → Host pool |
| 10 | 定位 RowBucketGraphRunner 调用者 | 指出模型专属路径与 warmup/地址稳定约束 |

前 3 个 demo 是概念实验，5..10 将概念映射回当前源码和真实服务。真实 tokenizer 长度以编码结果为准；练习 5 的数字是手算设定，不是直接按字符数构造 HTTP 输入。

## 对照源码

| 文件 | 读什么 |
|---|---|
| [sglang-learning-docs/06_demo_scheduler.py](../06_demo_scheduler.py) | 玩具队列 |
| [sglang-learning-docs/06_demo_radix_cache.py](../06_demo_radix_cache.py) | 玩具前缀 |
| [sglang-learning-docs/06_demo_zmq_pipeline.py](../06_demo_zmq_pipeline.py) | 玩具消息流 |
| [python/sglang/srt/managers/scheduler.py](../../python/sglang/srt/managers/scheduler.py) | 真正 plan/调度 |
