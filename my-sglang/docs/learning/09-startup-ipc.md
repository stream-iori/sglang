# 标准 SRT（四）：启动与进程通信

[返回全局地图](README.md) · 上一页：[采样到响应](08-standard-output.md) · 下一页：[CUDA Graph](10-cuda-graph.md)

**先把运行资源准备好，才有后面的逐轮生成。** 本篇补总图的启动前置条件与请求跨进程路径，以仓库 `a5a7123f54` 的普通 Python HTTP 服务、单 tokenizer、TP=1 为起点。

## 图 S8：一个服务由哪些进程协作

```text
客户端
   | HTTP
   v
主进程：HTTP / Engine / TokenizerManager
   | token 请求，携带 rid                    ^ 按 rid 唤醒等待者
   | IPC                                    | IPC
   v                                        |
Scheduler 子进程 -----------------> Detokenizer 子进程
   |             输出 token / 状态             |
   |                                          +-- 增量解码
   +-- TpModelWorker / ModelRunner
   +-- 模型权重、KV 池、GPU 执行
```

| 名字 | 是什么 | 不是哪种边界 |
|---|---|---|
| Scheduler 进程 | 接请求、组批，并组织设备执行的进程 | 不只是一个单独的队列对象 |
| TpModelWorker | Scheduler 内组织模型工作的对象 | 名称有 Worker 不代表独立 HTTP 服务 |
| CUDA stream | 设备工作的顺序与依赖机制 | 不等于 Python 子进程 |
| SMG Worker | 网关看到的一个后端服务资源 | 不等于每一个内部 TP rank |

基础路径使用 ZMQ PUSH/PULL 通道传递进程间消息。多 tokenizer、Rust server 等路径会改变通信组织，本图只作为普通路径坐标。

证据：[Engine 的进程说明与启动](../../../python/sglang/srt/entrypoints/engine.py)、[Tokenizer IPC](../../../python/sglang/srt/managers/tokenizer_manager.py)、[Detokenizer IPC](../../../python/sglang/srt/managers/detokenizer_manager.py)。

## 图 S9：启动资源与运行循环分开看

```text
读取配置 / 建立通信端点 / 启动子进程
                         |
              各执行进程准备模型与设备资源
                         |
          权重加载、KV 池、Attention backend、可选 Graph
                         |
                    就绪协调
                         |
              收请求 -> 组批 -> forward -> 返回
                 ^                           |
                 +---------------------------+
```

这是依赖图，不是所有配置下的逐行初始化顺序。`wait_for_ready()` 等就绪机制用于协调子进程；创建进程成功不等于模型已经可推理。

| 启动建立 | 请求运行时更新 |
|---|---|
| 模型参数与设备执行资源 | 本轮 input_ids、positions、长度 |
| 内存池及容量约束 | 哪些 row/page 分给哪些请求 |
| 通信端点 | 带 rid 的请求与结果 |
| 已捕获的 Graph（若启用） | replay 使用的缓冲区内容 |

不要认为每次 decode 都重新创建 KV pool。通常是复用池，更新其中的占用和内容。

## 请求如何回到正确的 HTTP 等待者

```text
请求 A：rid=A -> token 请求 -> Scheduler -> 输出带 A -> 解码带 A -> 唤醒 A
请求 B：rid=B -> token 请求 -> Scheduler -> 输出带 B -> 解码带 B -> 唤醒 B
```

A、B 可以被组进同一批，也可能分别结束。结果必须依靠请求标识分发，不能用“哪个 HTTP 请求最先到，就拿下一个结果”代替匹配。

这与 request row 不同：rid 识别请求，row 定位活跃 KV 地址表；retract 重新准入可换 row，但仍是同一条逻辑请求。

## 看日志时先定位边界

| 现象 | 先查什么证据 | 尚不能直接下的结论 |
|---|---|---|
| 启动未就绪 | 哪个子进程还在初始化、权重和内存池日志、退出异常 | “HTTP 慢就是模型 forward 慢” |
| 请求已收到，尚无首 token | tokenize、发送/接收请求、waiting 与 prefill 记录 | “请求一定已到 GPU” |
| Scheduler 已产出结果，客户端未看到 | 输出发送、detokenizer、rid 分发与 HTTP 流 | “GPU 必须再算一次” |

上表是取证路线，不是故障结论。更多链路见[本地请求流程](../local-chat-completions-flow.md)。

自检：`TpModelWorker` 是否一定是独立进程？不是；必须看创建和调用位置，不能仅根据名称推断。
