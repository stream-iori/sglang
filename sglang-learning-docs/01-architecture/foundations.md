# 从启动到第一条响应

> 源码基准：learning `2aa70e3eb4`，已合入 origin/main `6fc8d9da32`（2026-10-10）。

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

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/launch_server.py](../../python/sglang/launch_server.py) | run_server |
| [python/sglang/srt/entrypoints/engine.py](../../python/sglang/srt/entrypoints/engine.py) | 进程装配 |
| [python/sglang/srt/managers/tokenizer_manager.py](../../python/sglang/srt/managers/tokenizer_manager.py) | 请求编码与返回 |
| [python/sglang/srt/managers/tp_worker.py](../../python/sglang/srt/managers/tp_worker.py) | ForwardBatch 构造 |
