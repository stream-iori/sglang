# 动手练习：玩具模型帮助理解，真实服务负责验证

> 源码基准：learning `2aa70e3eb4`，已合入 origin/main `6fc8d9da32`（2026-10-10）。

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

## 对照源码

| 文件 | 读什么 |
|---|---|
| [sglang-learning-docs/06_demo_scheduler.py](../06_demo_scheduler.py) | 玩具队列 |
| [sglang-learning-docs/06_demo_radix_cache.py](../06_demo_radix_cache.py) | 玩具前缀 |
| [sglang-learning-docs/06_demo_zmq_pipeline.py](../06_demo_zmq_pipeline.py) | 玩具消息流 |
| [python/sglang/srt/managers/scheduler.py](../../python/sglang/srt/managers/scheduler.py) | 真正 plan/调度 |
