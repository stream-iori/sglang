# Week 1 详细讲义：从请求到第一个 token

> 目标：5 天内把 SGLang 的主干跑通。先知道“谁调用谁、数据怎么走”，不要陷入 CUDA、量化、多卡细节。
>
> **配套资源**: [Transformer 基本架构](../05-reference/prerequisites.md#transformer-basics-sglang) | [动手实验 - ZMQ 流水线 demo](../04-practice/exercises.md#practice-zmq-pipeline-demo) | [FAQ: event_loop 变体](../05-reference/faq.md#faq-event-loop-variants)

## 一句话结论

| 本周问题 | 最短答案 |
|---|---|
| SGLang 服务怎么起来？ | `sglang serve` 解析参数后启动 HTTP Server，HTTP Server 内部创建 Engine，Engine 拉起 Scheduler/Detokenizer 子进程。 |
| 核心进程有哪些？ | 主进程：HTTP Server + Engine + TokenizerManager；子进程：Scheduler、DetokenizerManager。 |
| 一个请求怎么走？ | HTTP JSON -> `GenerateReqInput` -> tokenize -> `TokenizedGenerateReqInput` -> `Req` -> `ScheduleBatch` -> `ForwardBatch` -> token id -> text -> HTTP response。 |
| Week 1 不看什么？ | 不看 CUDA kernel、FlashInfer、量化、多卡、PD 分离、投机解码。 |

## 1. 总图

```text
Client
  |
  | HTTP /v1/chat/completions
  v
Main Process
+-------------------------------------------------------------+
| http_server.py                                              |
|   -> OpenAI serving layer                                   |
|   -> GenerateReqInput                                       |
|                                                             |
| Engine                                                      |
|   -> TokenizerManager                                       |
|      text/input_ids -> TokenizedGenerateReqInput            |
+---------------------------+---------------------------------+
                            |
                            | ZMQ
                            v
Scheduler Process
+-------------------------------------------------------------+
| scheduler.py                                                |
|   TokenizedGenerateReqInput -> Req                          |
|   waiting_queue/running_batch -> ScheduleBatch              |
|   ScheduleBatch -> model_worker.forward_batch_generation()  |
+---------------------------+---------------------------------+
                            |
                            | ZMQ BatchTokenIDOutput
                            v
Detokenizer Process
+-------------------------------------------------------------+
| detokenizer_manager.py                                      |
|   token ids -> text                                         |
+---------------------------+---------------------------------+
                            |
                            | BatchStrOutput
                            v
TokenizerManager -> HTTP response -> Client
```

## 2. Day-by-Day

| 天 | 目标 | 必读代码 | 产出 |
|---|---|---|---|
| Day 1 | 看懂启动链路 | `python/sglang/cli/main.py`, `python/sglang/cli/serve.py`, `python/sglang/srt/entrypoints/http_server.py`, `python/sglang/srt/entrypoints/engine.py` | 画出 `sglang serve` 到 Engine 的调用链 |
| Day 2 | 看懂进程角色 | `engine.py:Engine`, `_launch_subprocesses`, `_launch_scheduler_processes`, `_launch_detokenizer_subprocesses` | 列出主进程/子进程职责 |
| Day 3 | 看懂消息类型 | `python/sglang/srt/managers/io_struct.py` | 解释 4 个核心消息类 |
| Day 4 | 看懂 Scheduler 主循环 | `scheduler.py:event_loop_normal`, `process_input_requests`, `handle_generate_request`, `get_next_batch_to_run` | 画出 waiting/running/batch 状态变化 |
| Day 5 | 串起请求生命周期 | `tokenizer_manager.py:generate_request`, `scheduler.py:run_batch`, `process_batch_result`, `model_runner.py:forward` | 自己复述全链路，不看文档 |

## 3. 启动链路

```text
sglang serve ...
  -> python/sglang/cli/main.py:main()
  -> python/sglang/cli/serve.py:serve()
  -> sglang.launch_server.run_server()
  -> python/sglang/srt/entrypoints/http_server.py:launch_server()
  -> create TokenizerManager
  -> create Engine
  -> Engine._launch_subprocesses()
       -> Scheduler subprocess
       -> Detokenizer subprocess
```

关键证据：

| 文件 | 代码点 | 说明 |
|---|---|---|
| `python/sglang/cli/main.py` | `if args.subcommand == "serve"` | CLI 分发到 `sglang.cli.serve.serve` |
| `python/sglang/cli/serve.py` | `run_server(server_args)` | 标准 LLM Server 入口 |
| `python/sglang/srt/entrypoints/http_server.py` | `launch_server()` | HTTP 服务入口 |
| `python/sglang/srt/entrypoints/engine.py` | `class Engine` | 引擎入口，负责拉起子进程和 ZMQ |

## 4. 进程职责

| 角色 | 进程 | 主要职责 | Week 1 读法 |
|---|---|---|---|
| HTTP Server | 主进程 | 接收 OpenAI API 请求，返回 JSON/SSE | 只看路由和 `GenerateReqInput` 构造 |
| Engine | 主进程 | 管理生命周期、创建 ZMQ socket、启动子进程 | 只看 `__init__` 和 `_launch_*` |
| TokenizerManager | 主进程 | text <-> token ids，维护请求状态，等待输出 | 只看 `generate_request()` |
| Scheduler | 子进程 | 排队、组 batch、KV cache、调用模型执行 | 只看主循环和请求入队 |
| ModelRunner/worker | Scheduler 侧 | 把 batch 转成模型 forward | Week 1 只知道它存在 |
| DetokenizerManager | 子进程 | token ids -> text | 只看输入输出数据类 |

注意：TokenizerManager 在当前 Engine 注释里明确属于主进程，不要把它当成独立子进程。

## 5. 消息与数据结构

```text
HTTP JSON
  |
  v
GenerateReqInput
  |  TokenizerManager tokenize
  v
TokenizedGenerateReqInput
  |  Scheduler.handle_generate_request
  v
Req
  |  Scheduler.get_next_batch_to_run
  v
ScheduleBatch
  |  model worker / ModelRunner
  v
ForwardBatch
  |
  v
BatchTokenIDOutput
  |  DetokenizerManager
  v
BatchStrOutput
```

| 数据结构 | 文件 | 大白话 |
|---|---|---|
| `GenerateReqInput` | `managers/io_struct.py` | HTTP 层收到的原始请求，还可能是文本 |
| `TokenizedGenerateReqInput` | `managers/io_struct.py` | Tokenizer 处理后的请求，已经有 token ids |
| `Req` | `managers/schedule_batch.py` | Scheduler 内部请求对象，带缓存、状态、停止条件 |
| `ScheduleBatch` | `managers/schedule_batch.py` | Scheduler 选出来的一批请求 |
| `ForwardBatch` | `model_executor/forward_batch_info.py` | 真正喂给模型执行侧的 Tensor 形态 batch |
| `BatchTokenIDOutput` | `managers/io_struct.py` | Scheduler 输出的新 token id |
| `BatchStrOutput` | `managers/io_struct.py` | Detokenizer 输出的文本 |

## 6. Scheduler 主循环

源码核心在 `python/sglang/srt/managers/scheduler.py:event_loop_normal()`。

```text
while True:
  recv_reqs = request_receiver.recv_requests()
  process_input_requests(recv_reqs)

  batch = get_next_batch_to_run()

  if batch:
    result = run_batch(batch)
    process_batch_result(batch, result)
  else:
    on_idle()
```

| 步骤 | 函数 | 学习重点 |
|---|---|---|
| 收请求 | `request_receiver.recv_requests()` | 从 TokenizerManager/RPC 拿消息 |
| 分发请求 | `process_input_requests()` | 调 `_request_dispatcher`，生成内部动作 |
| 生成 Req | `handle_generate_request()` | `TokenizedGenerateReqInput -> Req` |
| 选 batch | `get_next_batch_to_run()` | prefill/decode/chunked prefill 的主入口 |
| 执行 | `run_batch()` | 调模型 worker，返回 token/logprob/embedding |
| 处理结果 | `process_batch_result()` | decode/extend 分流，输出给 Detokenizer |

## 7. Prefill 和 Decode

| 阶段 | 触发场景 | 算什么 | 性能瓶颈 |
|---|---|---|---|
| Prefill/Extend | 新请求刚进来 | 整段 prompt | 计算量大，吞吐看并行 |
| Decode | 已经读完 prompt，继续生成 | 每轮 1 个或少量 token | 访存和调度更关键 |

```text
请求: "北京今天..."

Prefill:
  一口气读完整个 prompt，写入 KV Cache

Decode:
  第 1 轮 -> 生成 token A
  第 2 轮 -> 生成 token B
  第 3 轮 -> 生成 token C
```

Week 1 只需要记住：`ScheduleBatch.forward_mode` 会告诉系统本轮是 prefill/extend 还是 decode。

## 8. 动手命令

```bash
# 1. 找 CLI 入口
rg -n "def main|subcommand == \"serve\"|def serve|run_server" python/sglang/cli python/sglang

# 2. 找 Engine 启动子进程
rg -n "class Engine|def _launch_subprocesses|def _launch_scheduler_processes|def _launch_detokenizer_subprocesses" \
  python/sglang/srt/entrypoints/engine.py

# 3. 找核心消息类
rg -n "^class (GenerateReqInput|TokenizedGenerateReqInput|BatchTokenIDOutput|BatchStrOutput)" \
  python/sglang/srt/managers/io_struct.py

# 4. 找 Scheduler 主路径
rg -n "def event_loop_normal|def process_input_requests|def handle_generate_request|def get_next_batch_to_run|def run_batch|def process_batch_result" \
  python/sglang/srt/managers/scheduler.py
```

## 9. 本周验收

| 验收项 | 合格标准 |
|---|---|
| 启动链路 | 能从 `sglang serve` 说到 `Engine._launch_subprocesses()` |
| 进程拓扑 | 能说清主进程里有 HTTP/Engine/TokenizerManager，Scheduler/Detokenizer 是子进程 |
| 数据结构 | 能说清 `GenerateReqInput`、`TokenizedGenerateReqInput`、`Req`、`ScheduleBatch` 的区别 |
| 主循环 | 能默写 `recv -> process -> schedule -> run -> process_result` |
| 请求生命周期 | 能画出请求从 HTTP 到返回文本的全链路 |

## 10. 常见误区

| 误区 | 正解 |
|---|---|
| 一上来读完整个 `scheduler.py` | 先读 `event_loop_normal()`，再顺藤摸瓜 |
| 把 TokenizerManager 当子进程 | 当前 Engine 注释说明它和 HTTP/Engine 在主进程 |
| Week 1 研究 CUDA Graph | 跳过，Week 3/5 再看 |
| 分不清 `Req` 和 `GenerateReqInput` | 前者是 Scheduler 内部对象，后者是 HTTP/TokenizerManager 入口对象 |
| 觉得 ZMQ 很神秘 | 先理解成跨进程 Queue，细节以后补 |
