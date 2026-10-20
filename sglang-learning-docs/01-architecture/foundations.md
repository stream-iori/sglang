# 从启动到第一条响应

> 源码基准：上游 `16eb87c06f`；本地集成 `4f0e1cbea2`（2026-10-20）。本地日志增强与验证范围见总索引和 Mac 验证记录。

先把一次请求追完整，再研究优化。Python 主线仍在，但配置解析、缓存和执行器都已拆出新的层次。

## 启动的先后关系

```text
sglang.launch_server
  ├─ load_plugins / prepare_server_args
  ├─ resolve_once → 根据解析后的配置选择服务入口
  └─ HTTP / Engine 启动
       ├─ TokenizerManager
       ├─ scheduler 子进程：加载权重 → KV pools → Attention → Graph/预热
       ├─ DetokenizerManager 子进程
       └─ 服务就绪
```

启动日志出现“加载模型”不等于请求可用了。就绪还涉及池初始化、预热和服务入口。

| 文件 | 角色 | 第一次阅读找什么 |
|---|---|---|
| launch_server.py | 入口 | run_server 的分支与 resolving_view |
| server_args.py / arg_groups | 参数声明与解析 | 原始输入怎样变成有效配置 |
| entrypoints/engine.py | 进程装配 | _launch_subprocesses、scheduler/detokenizer 启动 |
| managers/tokenizer_manager.py | HTTP 与引擎桥梁 | generate_request，rid 和异步返回 |
| managers/scheduler.py | 批次选择 | event_loop_normal |
| managers/tp_worker.py | 调度到模型的桥梁 | ForwardBatch.init_new |

## 一条普通 generate 请求

```text
{"text":"Hello", "sampling_params":{"max_new_tokens":8}}
    ↓ 编码
GenerateReqInput → TokenizedGenerateReqInput
    ↓ 跨进程消息
Req → waiting_queue → EXTEND → running_batch → DECODE
    ↓ 每次生成一个或多个 ID
输出消息 → 增量解码 → HTTP 响应或 SSE 片段
```

OpenAI chat 会先应用模型的 chat template。`messages` 不是直接输入模型的张量。

## 第一轮观察

```bash
bash sglang-learning-docs/setup/launch_mac.sh
# 另一终端
python/.venv/bin/python sglang-learning-docs/setup/verify_mac.py
```

验收：能指出请求在哪个进程编码、在哪个进程排队、在哪个对象里维护输出 ID。实际成功记录见 [Mac 验证](../setup/mac-validation.md)。

## 按进程追请求，不按文件列表背架构

| 位置 | 输入 | 输出 | 第一个观察点 |
|---|---|---|---|
| HTTP / TokenizerManager | text 或 messages | token IDs、rid、采样参数 | generate_request、请求编码 |
| Scheduler 进程 | tokenized 请求 | Req、每轮计划、输出 ID 消息 | waiting_queue、run_batch |
| worker / runner | 本轮 token、长度和地址 | logits、采样结果 | ForwardBatch.init_new、_forward_raw |
| DetokenizerManager 进程 | 累计或增量 ID | 文本片段和结束标志 | 解码偏移与输出去重 |

它们通过 IPC 连接；HTTP 协程等待响应并不等于这个请求独占模型。多个 HTTP 请求可在同一个调度器中轮流或合批执行。

## 把启动失败分成四段

```text
参数/依赖检查 → 子进程和 IPC → 权重/池/Attention → 预热/health
     A              B                  C                  D
```

| 现象 | 优先证据 | 下一步 |
|---|---|---|
| 立刻退出 | 主进程 traceback、包版本 | 查导入、参数解析、MPS runtime |
| 子进程未就绪 | 首个子进程错误与退出码 | 查 scheduler/IPC 初始化，不能只看 HTTP 等待 |
| 权重已加载但不能请求 | KV、Attention、warmup 日志 | 查后续初始化失败或尚未完成 |
| health 200 但生成失败 | 请求响应和同 rid 日志 | 转到调度、forward、输出路径 |

## 第一份可复现记录

先按 Mac 指南选择环境，以下命令假定默认 `python/.venv` 已安装当前依赖：

```bash
PYTHONPATH=python python/.venv/bin/python -c 'import sglang; print(sglang.__file__)'
python/.venv/bin/python -c 'import torch; print(torch.__version__, torch.backends.mps.is_available())'
```

打印路径应落在本仓库的 `python/sglang/`。若指向其他 site-packages，先修正导入来源，再研究代码行为。

验收：说明 `/health`、模型成功生成、缓存命中分别证明哪一层可用。答案：它们分别覆盖服务就绪、执行与返回、重复输入复用；三项不能互相替代。

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/launch_server.py](../../python/sglang/launch_server.py) | run_server |
| [python/sglang/srt/entrypoints/engine.py](../../python/sglang/srt/entrypoints/engine.py) | 进程装配 |
| [python/sglang/srt/managers/tokenizer_manager.py](../../python/sglang/srt/managers/tokenizer_manager.py) | 请求编码与返回 |
| [python/sglang/srt/managers/tp_worker.py](../../python/sglang/srt/managers/tp_worker.py) | ForwardBatch 构造 |
