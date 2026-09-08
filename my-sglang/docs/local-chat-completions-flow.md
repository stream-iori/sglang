# 以 SMG 为入口：本地无 PD 的 `/v1/chat/completions` 全链路

> 代码基线：`2fb08837a82d3876e9c818eae91dab44d852e043`。行号对应当前 checkout。

## 先说结论

本地无 PD、以 SMG 为入口时，请求会经过两层同名接口：

```text
客户端
  │ POST SMG:/v1/chat/completions
  ▼
SMG（Rust）
  │ 选中一个 regular HTTP worker
  │ POST SRT-Worker:/v1/chat/completions
  ▼
SRT（Python）
  │ chat template -> tokenize -> prefill -> decode -> detokenize
  ▼
SMG
  │ 原样转发 JSON body 或 SSE byte stream
  ▼
客户端
```

| 层级 | “无 PD”在这一层的含义 | 代码证据 |
|---|---|---|
| SMG | 不启用 `--pd-disaggregation`，创建 `RoutingMode::Regular`，一次请求只选一个 regular worker | [`main.rs:905-926`](../../sgl-model-gateway/src/main.rs#L905-L926) |
| SRT worker | `disaggregation_mode="null"`，同一个 unified Scheduler 完成 prefill 和 decode | [`server_args.py:3168-3172`](../../python/sglang/srt/server_args.py#L3168-L3172)、[`scheduler.py:2897-2919`](../../python/sglang/srt/managers/scheduler.py#L2897-L2919) |

因此：

```text
无 PD = SMG 不做 prefill/decode 双路转发
      + SRT 不在不同实例间传 KV

无 PD ≠ 没有 prefill 和 decode
```

## 1. 本文采用的本地拓扑

```text
┌────────────────────────────── 本机 ────────────────────────────────┐
│                                                                  │
│  Client                                                          │
│    │                                                             │
│    │ :19000                                                      │
│    ▼                                                             │
│  SMG / Axum                                                      │
│    │ worker URL = http://127.0.0.1:30000                         │
│    │                                                             │
│    ▼ :30000                                                      │
│  SRT HTTP 主进程                                                 │
│    ├── FastAPI / OpenAI serving                                  │
│    └── TokenizerManager                                          │
│            │ ZMQ                                                 │
│            ▼                                                     │
│       Scheduler 子进程 ── GPU / model / KV cache                 │
│            │ ZMQ                                                 │
│            ▼                                                     │
│       Detokenizer 子进程                                         │
│            │ ZMQ                                                 │
│            └──────────────> TokenizerManager                     │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

本文固定以下条件，避免混入其他路由：

| 项目 | 取值 |
|---|---|
| SMG backend | 默认 `sglang` |
| SMG PD | 不传 `--pd-disaggregation` |
| SMG IGW | 不传 `--enable-igw` |
| SMG connection | `http://...` worker URL |
| SRT PD | 默认 `disaggregation_mode=null` |
| SRT TP / PP / DP | 主线按 `1 / 1 / 1` |
| 请求 | 文本 chat、`n=1`、无多模态、无 speculative、无 beam search |

SMG 的 `enable_igw`、`pd_disaggregation` 和 `dp_aware` 默认都是 `false`，见 [`main.rs:188-200`](../../sgl-model-gateway/src/main.rs#L188-L200)；默认 backend 是 SGLang，见 [`main.rs:469-472`](../../sgl-model-gateway/src/main.rs#L469-L472)。

## 2. 启动阶段：SMG 如何进入 regular 模式

### 2.1 配置选择

```text
CLI 参数
  │
  ├─ backend == openai ? ──是──> RoutingMode::OpenAI
  │
  ├─ pd_disaggregation ? ──是──> RoutingMode::PrefillDecode
  │
  └─ 否 ───────────────────────> RoutingMode::Regular { worker_urls }
                                      ↑
                                      └── 本文路径
```

选择逻辑在 [`main.rs:905-926`](../../sgl-model-gateway/src/main.rs#L905-L926)。CLI 随后构造 RouterConfig、ServerConfig 和 Tokio runtime，并调用 `server::startup()`，见 [`main.rs:1189-1275`](../../sgl-model-gateway/src/main.rs#L1189-L1275)。

### 2.2 注册 worker 与创建 router

```text
server::startup
      │
      ├─ AppContext::from_config
      │
      ├─ Job::InitializeWorkersFromConfig
      │        │
      │        └─ Regular.worker_urls
      │             -> 每个 URL 创建 regular AddWorker job
      │             -> 探测并注册到 WorkerRegistry
      │
      ├─ RouterManager::from_config
      │        │ enable_igw=false
      │        └─ RouterFactory::create_router
      │             -> HTTP Regular Router
      │             -> 设置为 default_router
      │
      └─ build_app -> 绑定端口 -> Axum serve
```

| 动作 | 代码位置 |
|---|---|
| 创建 AppContext 和 worker job queue | [`server.rs:801-834`](../../sgl-model-gateway/src/server.rs#L801-L834) |
| 提交配置中的 worker 初始化任务 | [`server.rs:876-894`](../../sgl-model-gateway/src/server.rs#L876-L894) |
| `RoutingMode::Regular` 把 URL 转成 regular worker | [`job_queue.rs:496-519`](../../sgl-model-gateway/src/core/job_queue.rs#L496-L519) |
| 为每个 URL 提交 `AddWorker` | [`job_queue.rs:596-630`](../../sgl-model-gateway/src/core/job_queue.rs#L596-L630) |
| 单 router 模式创建并设置默认 router | [`router_manager.rs:81-200`](../../sgl-model-gateway/src/routers/router_manager.rs#L81-L200) |
| RouterManager 成为 `AppState.router` | [`server.rs:917-924`](../../sgl-model-gateway/src/server.rs#L917-L924)、[`server.rs:992-999`](../../sgl-model-gateway/src/server.rs#L992-L999) |
| 构建 Axum app 并绑定监听地址 | [`server.rs:1028-1068`](../../sgl-model-gateway/src/server.rs#L1028-L1068) |

## 3. 完整触发流程

图中 `G1..G8` 是 SMG，`S1..S9` 是 SRT。

```text
Client
  │
  │ POST /v1/chat/completions
  │ ChatCompletionRequest { model, messages, stream, ... }
  ▼
┌──────────────────────────── SMG / Rust ─────────────────────────────┐
│ Axum route + ValidatedJson [G1]                                    │
│   │                                                               │
│   ▼                                                               │
│ RouterManager::route_chat [G2]                                    │
│   │ single-router: 直接取 default HTTP Regular Router             │
│   ▼                                                               │
│ http::Router::route_chat                                           │
│   -> route_typed_request [G3]                                     │
│   │                                                               │
│   ├─ 提取 stream、model、用于路由的 text                          │
│   ├─ RetryExecutor：失败时可能重新执行一次完整 attempt             │
│   ▼                                                               │
│ route_typed_request_once                                           │
│   │                                                               │
│   ├─ WorkerRegistry：regular + HTTP 候选集                         │
│   ├─ is_available：健康检查 + circuit breaker                     │
│   ├─ LoadBalancingPolicy::select_worker                           │
│   ▼                                                     [G4]      │
│ send_typed_request                                                 │
│   ├─ worker.url + "/v1/chat/completions"                          │
│   ├─ RequestBuilder.json(typed_req)                               │
│   ├─ Authorization / 可转发 headers                               │
│   └─ request_builder.send().await                       [G5]      │
└───────────────────────────┬────────────────────────────────────────┘
                            │ HTTP POST
                            ▼
┌────────────────────── SRT Worker / Python ─────────────────────────┐
│ FastAPI /v1/chat/completions [S1]                                 │
│   ▼                                                               │
│ OpenAIServingChat                                                 │
│   messages -> chat template -> prompt text -> input_ids [S2]      │
│   ▼                                                               │
│ GenerateReqInput -> TokenizedGenerateReqInput                     │
│   │ ZMQ PUSH                                             [S3]     │
│   ▼                                                               │
│ Scheduler                                                         │
│   TokenizedGenerateReqInput -> Req -> waiting_queue               │
│   DisaggregationMode.NULL                               [S4]      │
│   │                                                               │
│   ├─ RadixCache.match_prefix                                     │
│   ├─ EXTEND / prefill：计算未缓存 prompt，写 KV，sample x0        │
│   └─ DECODE：每轮输入上一个 token，写 1 个 KV slot，sample xn     │
│          └─ 直到 EOS / stop / max_tokens                 [S5]     │
│   │                                                               │
│   ▼ SchedulerOutputStreamer                                       │
│ BatchTokenIDOutput                                      [S6]      │
│   │ ZMQ                                                           │
│   ▼                                                               │
│ DetokenizerManager：token IDs -> 增量文本                [S7]      │
│   │ ZMQ                                                           │
│   ▼                                                               │
│ TokenizerManager：按 rid 更新 ReqState，唤醒 HTTP 协程   [S8]      │
│   ▼                                                               │
│ ChatCompletionResponse JSON 或 SSE chunks + [DONE]       [S9]     │
└───────────────────────────┬────────────────────────────────────────┘
                            │ HTTP response
                            ▼
┌──────────────────────────── SMG / Rust ─────────────────────────────┐
│ send_typed_request 收到 reqwest::Response [G6]                    │
│   │                                                               │
│   ├─ stream=false：读完整 bytes，复制 status/headers/body [G7]   │
│   └─ stream=true ：bytes_stream 直接包装为 Axum Body       [G8]   │
│                      BreakerTrackedStream 跟踪结束/错误             │
└───────────────────────────┬────────────────────────────────────────┘
                            ▼
                          Client
```

### 源码索引

| 编号 | 触发点 | 代码位置 |
|---|---|---|
| G1 | 注册 SMG 路由；handler 把 JSON 反序列化为 `ChatCompletionRequest` | [`server.rs:184-202`](../../sgl-model-gateway/src/server.rs#L184-L202)、[`server.rs:545-556`](../../sgl-model-gateway/src/server.rs#L545-L556) |
| G2 | RouterManager 选择 router，再调用其 `route_chat` | [`router_manager.rs:527-568`](../../sgl-model-gateway/src/routers/router_manager.rs#L527-L568) |
| G3 | HTTP Regular Router 把 chat 固定映射回 `/v1/chat/completions`，外层包重试 | [`router.rs:206-283`](../../sgl-model-gateway/src/routers/http/router.rs#L206-L283)、[`router.rs:790-798`](../../sgl-model-gateway/src/routers/http/router.rs#L790-L798) |
| G4 | 查 regular HTTP worker、过滤不可用 worker、调用 policy 选下标 | [`router.rs:133-203`](../../sgl-model-gateway/src/routers/http/router.rs#L133-L203) |
| G5 | 构造 upstream RequestBuilder、转发 header、发送请求 | [`router.rs:498-610`](../../sgl-model-gateway/src/routers/http/router.rs#L498-L610) |
| G6 | 取得上游 status | [`router.rs:612-623`](../../sgl-model-gateway/src/routers/http/router.rs#L612-L623) |
| G7 | 非流式读取完整 body 并构造 Axum Response | [`router.rs:625-643`](../../sgl-model-gateway/src/routers/http/router.rs#L625-L643) |
| G8 | 流式把 reqwest byte stream 包成响应体，并附加 breaker/load guard | [`router.rs:644-680`](../../sgl-model-gateway/src/routers/http/router.rs#L644-L680) |
| S1 | SRT worker 注册同名 FastAPI 路由 | [`http_server.py:1733-1740`](../../python/sglang/srt/entrypoints/http_server.py#L1733-L1740) |
| S2 | OpenAI 请求转内部请求；应用 chat template 并编码 token | [`serving_base.py:73-109`](../../python/sglang/srt/entrypoints/openai/serving_base.py#L73-L109)、[`serving_chat.py:966-1098`](../../python/sglang/srt/entrypoints/openai/serving_chat.py#L966-L1098)、[`serving_chat.py:1385-1396`](../../python/sglang/srt/entrypoints/openai/serving_chat.py#L1385-L1396) |
| S3 | TokenizerManager 创建状态、tokenized 对象并经 ZMQ 发送 | [`tokenizer_manager.py:770-836`](../../python/sglang/srt/managers/tokenizer_manager.py#L770-L836)、[`tokenizer_manager.py:1336-1450`](../../python/sglang/srt/managers/tokenizer_manager.py#L1336-L1450)、[`tokenizer_manager.py:1557-1577`](../../python/sglang/srt/managers/tokenizer_manager.py#L1557-L1577) |
| S4 | Scheduler 创建 Req；NULL 模式加入 unified waiting queue | [`scheduler.py:2487-2555`](../../python/sglang/srt/managers/scheduler.py#L2487-L2555)、[`scheduler.py:2897-2919`](../../python/sglang/srt/managers/scheduler.py#L2897-L2919) |
| S5 | 选择 prefill/decode batch，执行模型并处理 sampled token | [`scheduler.py:3193-3341`](../../python/sglang/srt/managers/scheduler.py#L3193-L3341)、[`scheduler.py:3863-4041`](../../python/sglang/srt/managers/scheduler.py#L3863-L4041)、[`batch_result_processor.py:869-1008`](../../python/sglang/srt/managers/scheduler_components/batch_result_processor.py#L869-L1008) |
| S6 | 聚合可输出 token，生成 `BatchTokenIDOutput` | [`output_streamer.py:118-215`](../../python/sglang/srt/managers/scheduler_components/output_streamer.py#L118-L215)、[`output_streamer.py:690-750`](../../python/sglang/srt/managers/scheduler_components/output_streamer.py#L690-L750) |
| S7 | Detokenizer 增量解码 | [`detokenizer_manager.py:301-420`](../../python/sglang/srt/managers/detokenizer_manager.py#L301-L420)、[`detokenizer_manager.py:441-509`](../../python/sglang/srt/managers/detokenizer_manager.py#L441-L509) |
| S8 | 按 rid 更新 ReqState，并通过 event 唤醒请求协程 | [`tokenizer_manager.py:2187-2495`](../../python/sglang/srt/managers/tokenizer_manager.py#L2187-L2495) |
| S9 | 构造流式或非流式 OpenAI 响应 | [`serving_chat.py:1515-1807`](../../python/sglang/srt/entrypoints/openai/serving_chat.py#L1515-L1807)、[`serving_chat.py:1809-2011`](../../python/sglang/srt/entrypoints/openai/serving_chat.py#L1809-L2011) |

## 4. SMG 收到请求后做了什么

### 4.1 Axum 只负责接住请求

```text
POST /v1/chat/completions
          │
          ▼
ValidatedJson<ChatCompletionRequest>
          │
          ▼
state.router.route_chat(headers, body, body.model)
```

handler 不选 worker，也不运行模型；它只把强类型请求交给 `RouterTrait`。路由注册在 [`server.rs:545-556`](../../sgl-model-gateway/src/server.rs#L545-L556)，handler 在 [`server.rs:184-202`](../../sgl-model-gateway/src/server.rs#L184-L202)。

### 4.2 为什么无 PD 会落到 HTTP Regular Router

在本文的 `enable_igw=false` 单 router 模式中，RouterManager 总是取启动时设置的 `default_router`：

```text
RoutingMode::Regular + ConnectionMode::Http
                  │
                  ▼
          router id = http-regular
                  │
                  ▼
          default_router
                  │
                  ▼
        RouterManager::route_chat
```

router ID 映射在 [`router_manager.rs:189-200`](../../sgl-model-gateway/src/routers/router_manager.rs#L189-L200)，单 router 选择在 [`router_manager.rs:328-347`](../../sgl-model-gateway/src/routers/router_manager.rs#L328-L347)。因此本路径不会进入 `http/pd_router.rs`。

### 4.3 如何选出一个 worker

```text
WorkerRegistry
     │ get_workers_filtered(
     │   worker_type = Regular,
     │   connection  = Http)
     ▼
候选 workers
     │ is_available()
     ▼
健康且 circuit 未打开的 workers
     │ policy.select_worker(...)
     ▼
Arc<dyn Worker>
```

代码在 [`router.rs:133-203`](../../sgl-model-gateway/src/routers/http/router.rs#L133-L203)。

一个容易误读的细节：未开启 IGW 时，`effective_model_id=None`，registry 不按请求中的 model 过滤；但 `model_id` 仍用于取得对应 policy。也就是说：

```text
未开 IGW：model -> 选择 policy
           worker 候选 -> regular + HTTP
```

### 4.4 `send_typed_request` 如何生成真正的上游请求

默认 `dp_aware=false` 时走 `else`：

```rust
let mut request_builder = if self.dp_aware {
    // DP-aware：改写 URL，并向 JSON 加 data_parallel_rank
    ...
} else {
    self.client
        .post(format!("{}{}", worker_url, route))
        .json(typed_req)
};
```

源码：[`router.rs:514-560`](../../sgl-model-gateway/src/routers/http/router.rs#L514-L560)。假设：

```text
worker_url = http://127.0.0.1:30000
route      = /v1/chat/completions

最终 URL   = http://127.0.0.1:30000/v1/chat/completions
```

`if` 在 Rust 中是表达式。两个分支都产生 `reqwest::RequestBuilder`，整个表达式结束后才绑定给 `request_builder`：

```text
if self.dp_aware 的某个分支
          │ 产生 RequestBuilder
          ▼
let mut request_builder = ...
          │
          ├─ 添加 worker API key
          ├─ 添加允许转发的客户端 headers
          └─ send().await
```

`request_builder` 不能在初始化它的 `if self.dp_aware {}` 分支中使用；它只能从语句末尾 [`router.rs:560`](../../sgl-model-gateway/src/routers/http/router.rs#L560) 之后使用。后续 header 和发送逻辑见 [`router.rs:562-610`](../../sgl-model-gateway/src/routers/http/router.rs#L562-L610)。

### 4.5 重试边界

`route_typed_request()` 用 `RetryExecutor` 包住整个 `route_typed_request_once()`：

```text
attempt N
  -> 重新 select_worker_for_model
  -> 重新 send_typed_request
  -> 得到 Response
  -> 状态可重试？
       ├─ 是：backoff 后进入 attempt N+1
       └─ 否：返回
```

源码：[`router.rs:206-283`](../../sgl-model-gateway/src/routers/http/router.rs#L206-L283)。因此日志中 worker selection 和 upstream send 可以出现多次；不是同一个 RequestBuilder 被重复 `send()`。

## 5. 请求进入本地 SRT worker 后做了什么

### 5.1 SRT 并不是单进程

SMG 只把 SRT 看成一个 HTTP worker URL；SRT 内部默认仍有三部分：

```text
SRT 主进程
  FastAPI + OpenAIServingChat + TokenizerManager
                           │
                           │ ZMQ
                           ▼
Scheduler 子进程：调度、模型 forward、KV cache、sampling
                           │
                           │ ZMQ
                           ▼
Detokenizer 子进程：token IDs -> text
                           │
                           └── ZMQ -> TokenizerManager
```

SRT 启动入口会创建 Scheduler 和 Detokenizer 子进程，见 [`http_server.py:2796-2824`](../../python/sglang/srt/entrypoints/http_server.py#L2796-L2824)、[`engine.py:820-966`](../../python/sglang/srt/entrypoints/engine.py#L820-L966)。

### 5.2 `messages` 如何变成 token

```text
ChatCompletionRequest
  {messages, temperature, max_tokens, stream, ...}
          │ OpenAIServingChat
          ▼
chat template 渲染后的 prompt text
          │ tokenizer.encode
          ▼
GenerateReqInput { input_ids, sampling_params, ... }
          │ TokenizerManager
          ▼
TokenizedGenerateReqInput
          │ ZMQ
          ▼
Scheduler Req
```

默认 Jinja 路径调用 `apply_chat_template(..., tokenize=False, add_generation_prompt=True)`，然后 `tokenizer.encode`，见 [`serving_chat.py:1331-1396`](../../python/sglang/srt/entrypoints/openai/serving_chat.py#L1331-L1396)。Chat serving 已给出 `input_ids` 时，TokenizerManager 直接复用，不会再次 encode，见 [`tokenizer_manager.py:961-1000`](../../python/sglang/srt/managers/tokenizer_manager.py#L961-L1000)。

### 5.3 unified Scheduler 中的 prefill 与 decode

```text
TokenizedGenerateReqInput
          │
          ▼
Req -> waiting_queue                 DisaggregationMode.NULL
          │
          ▼
RadixCache.match_prefix
          │
          ▼
EXTEND / prefill
  只计算未命中的 prompt tokens
  写 prompt KV
  sample 第一个输出 token x0
          │
          ▼
DECODE #1：输入 x0，新增一个 KV slot，sample x1
          │
          ▼
DECODE #2：输入 x1，新增一个 KV slot，sample x2
          │
          └── 重复到 EOS / stop / max_tokens
```

| 阶段 | 代码位置 |
|---|---|
| NULL 模式加入 unified waiting queue | [`scheduler.py:2897-2919`](../../python/sglang/srt/managers/scheduler.py#L2897-L2919) |
| 选择新 prefill batch 或 running decode batch | [`scheduler.py:3193-3341`](../../python/sglang/srt/managers/scheduler.py#L3193-L3341) |
| 前缀匹配、组 prefill batch | [`scheduler.py:3390-3647`](../../python/sglang/srt/managers/scheduler.py#L3390-L3647) |
| EXTEND 只保留未缓存输入 | [`schedule_batch.py:2445-2492`](../../python/sglang/srt/managers/schedule_batch.py#L2445-L2492) |
| DECODE 每请求准备一个新 token 位置 | [`schedule_batch.py:3180-3227`](../../python/sglang/srt/managers/schedule_batch.py#L3180-L3227) |
| ModelRunner forward 与 sampler | [`tp_worker.py:593-703`](../../python/sglang/srt/managers/tp_worker.py#L593-L703)、[`model_runner.py:1555-1649`](../../python/sglang/srt/model_executor/model_runner.py#L1555-L1649)、[`model_runner.py:1827-1881`](../../python/sglang/srt/model_executor/model_runner.py#L1827-L1881) |
| 追加 token、判断结束 | [`batch_result_processor.py:869-1008`](../../python/sglang/srt/managers/scheduler_components/batch_result_processor.py#L869-L1008)、[`schedule_batch.py:1676-1716`](../../python/sglang/srt/managers/schedule_batch.py#L1676-L1716) |

### 5.4 token 如何变回 OpenAI 响应

```text
Req.output_ids
      │ SchedulerOutputStreamer
      ▼
BatchTokenIDOutput
      │ ZMQ
      ▼
DetokenizerManager
      ▼
BatchStrOutput
      │ ZMQ
      ▼
TokenizerManager.rid_to_state[rid]
      │ event.set()
      ▼
OpenAIServingChat
      ├─ stream=false -> ChatCompletionResponse JSON
      └─ stream=true  -> SSE chunks -> data: [DONE]
```

## 6. SRT 返回后，SMG 是否修改响应

主路径不解析 completion 内容：

| 模式 | SMG 行为 | 代码位置 |
|---|---|---|
| `stream=false` | 等待上游完整 body，复制允许保留的 headers 和 status，用相同 bytes 构造 Axum Response | [`router.rs:625-643`](../../sgl-model-gateway/src/routers/http/router.rs#L625-L643) |
| `stream=true` | 不等待完整 SSE；直接把 `res.bytes_stream()` 包成 Axum Body | [`router.rs:644-680`](../../sgl-model-gateway/src/routers/http/router.rs#L644-L680) |

流式路径：

```text
SRT SSE bytes
      │ reqwest::Response::bytes_stream()
      ▼
BreakerTrackedStream
      │ 区分正常结束 / 上游错误 / 客户端断开
      │ 客户端断开时不记录 worker 成功或失败
      ▼
Axum Body
      ▼
Client
```

所以 SMG 的 `upstream_response_headers` 日志只代表 SRT 响应头已到达，不代表流式生成已经结束。

## 7. 一次请求的时序图

```text
Client       SMG Axum     RouterManager     HTTP Router       SRT Worker
  │              │              │               │                 │
  │ POST /v1/chat/completions    │               │                 │
  ├─────────────>│              │               │                 │
  │              │ route_chat   │               │                 │
  │              ├─────────────>│               │                 │
  │              │              │ default router│                 │
  │              │              ├──────────────>│                 │
  │              │              │               │ select worker   │
  │              │              │               │ build request   │
  │              │              │               │ POST same route │
  │              │              │               ├────────────────>│
  │              │              │               │                 │ template/tokenize
  │              │              │               │                 │ prefill
  │              │              │               │                 │ decode * N
  │              │              │               │                 │ detokenize
  │              │              │               │ JSON/SSE        │
  │              │              │               │<────────────────┤
  │              │              │<──────────────┤                 │
  │              │<─────────────┤               │                 │
  │ JSON/SSE     │              │               │                 │
  │<─────────────┤              │               │                 │
```

## 8. 最短排查路径

| 现象 | 第一观察点 | 下一跳 |
|---|---|---|
| 请求没进 SMG | `server.rs:v1_chat_completions` | Axum 路由、中间件、鉴权 |
| SMG 返回找不到 router | `RouterManager::route_chat` | `RoutingMode`、`default_router` |
| SMG 返回没有可用 worker | `select_worker_for_model` | WorkerRegistry、健康状态、circuit breaker |
| 不知道请求发到哪里 | `send_typed_request` 的 `worker_url + route` | worker 注册 URL |
| SRT 收不到请求 | SMG `request_builder.send()` 与 SRT FastAPI route | URL、端口、HTTP 状态 |
| SRT 收到但不出 token | Scheduler waiting queue -> prefill batch -> ModelRunner | 显存、调度、模型 forward |
| SRT 已出 token 但 SMG 没返回 | Detokenizer -> TokenizerManager -> SRT HTTP response | 非流式 body / SSE stream |
| 流式中途断开 | `BreakerTrackedStream` | 上游 stream error 或 client disconnect |

## 9. 继续阅读

- [从 SGLang 运行时到 SGL Model Gateway：学习路线](smg-learning-path.md)
- [用 Fake Worker 本地调试 SMG](smg-fake-worker-lab.md)
- [Scheduler 与 KV 概览](scheduler-kv-overview.md)
- [模型执行连接层](model-execution-bridge.md)
- [Overlap Pipeline](overlap-pipeline.md)
