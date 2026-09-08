# 从 SGLang 运行时到 SGL Model Gateway：学习路线

先看 [SMG 能力边界与核心模块](smg-capability-map.md)，了解数据面、控制面与 SRT 的分工，再按本文逐段学习。

这份路线面向已经理解 `my-sglang` 基础主线、并有基本 Rust 阅读能力的读者。目标不是
立刻掌握 SMG 的所有 API，而是能从一次请求解释它为什么选中一个 worker、如何把请求可靠地
转发出去，以及 worker 失效或拓扑改变后系统如何反应。

`my-sglang` 与 SGL Model Gateway（SMG）处在不同层：前者解释**一个 worker 内部**怎样调度
请求、维护 KV，并计算下一个 token；后者解释**多个 worker 之间**怎样注册、观测、选择和
转发请求。

```text
my-sglang / 单个 SGLang worker
request -> prefill/decode -> KV/radix -> token

SMG / 多个 worker 的控制面和数据面
request -> router -> policy -> selected worker -> upstream response
```

因此，不要尝试把 `MiniScheduler` 改造成网关。学习 SMG 时，把 `my-sglang` 当作理解
worker 内部负载、KV cache、prefill/decode 和 radix prefix 的前置知识。

## 先建立最小心智模型

先只看常规 HTTP 请求，忽略 gRPC、PD、MCP、Mesh 和持久化。一次请求的最小路径是：

```text
Client
  -> Axum endpoint
  -> Router
  -> WorkerRegistry 取得候选 worker
  -> LoadBalancingPolicy 选择一个健康 worker
  -> HTTP 转发到 SGLang worker
  -> JSON 或 SSE streaming response 回到 Client
```

这里有三个核心对象：

| 对象 | 它回答的问题 | 对应源码 |
|---|---|---|
| `WorkerRegistry` | 现在有哪些 worker，它们是否健康、属于哪个模型/类型？ | [`core/worker_registry.rs`](../../sgl-model-gateway/src/core/worker_registry.rs) |
| `LoadBalancingPolicy` | 这条请求应该选择哪个候选 worker？ | [`policies/mod.rs`](../../sgl-model-gateway/src/policies/mod.rs) |
| Router | 选中后如何转发请求、处理 streaming、错误和协议差异？ | [`routers/`](../../sgl-model-gateway/src/routers/) |

`AppContext` 把这些对象及 tokenizer、限流、存储、MCP、WASM 和观测组件装配到一个运行时中。
读源码时可把它当作全局依赖图，而不是业务逻辑入口。

## Rust 前置知识：够用即可

无需先学完 Rust 异步生态；在开始前能够读懂下列概念即可：

- `async`/`await` 和 Tokio task；
- `Result`、`?` 和自定义错误；
- `Arc<T>`、`Arc<dyn Trait>`、`Send + Sync`；
- Axum 的 `State`、handler 与 middleware；
- `serde`/`serde_json` 的请求数据；
- SSE streaming 和 HTTP header 的基本语义。

SMG 用 Axum/Tokio 承载 HTTP 服务，以 `Arc<dyn Worker>` 和 trait 抽象 worker/policy；这些
概念比 Rust 的全部语法细节更优先。

## 阶段 1：跑通普通 HTTP 路由

先按 [`sgl-model-gateway/README.md`](../../sgl-model-gateway/README.md) 启动一个 router 与
一个或两个 SGLang worker，发送一次 `/v1/chat/completions` 请求。开始时只使用
`round_robin` 或 `random` 策略。

此阶段的目标是能观察并解释：

```text
HTTP request -> selected worker URL -> upstream response -> client response
```

不要先上 cache-aware、gRPC、PD 或 MCP；它们会同时引入 tokenization、双 worker 配对、状态机
或工具循环，使最简单的路由主线不再可见。

不想启动 GPU 或下载模型时，先完成[用 Fake Worker 本地调试 SMG](smg-fake-worker-lab.md)的
“实验一”。它会启动一个独立的协议模拟 worker，让这条最小路径可在本机观察。

## 阶段 2：按一次请求读源码

不要从很长的 CLI 参数定义开始。按下面顺序跟踪一条 `/v1/chat/completions`：

需要从 SMG 入口继续跟进本地 SRT worker 内部的 prefill、decode 和返回链路时，直接看
[以 SMG 为入口：本地无 PD 的 `/v1/chat/completions` 全链路](local-chat-completions-flow.md)。

这一步只证明一件事：**一条 chat 请求如何变成一次到某个 HTTP worker 的转发。**先不读 PD、
gRPC、MCP，也不改业务代码。

```text
POST /v1/chat/completions  (x-request-id=learn-chat-001)
  -> server::v1_chat_completions
  -> RouterManager::route_chat
  -> http::Router::select_worker_for_model
  -> LoadBalancingPolicy::select_worker
  -> http::Router::send_typed_request
  -> selected worker URL + /v1/chat/completions
```

| 想确认的事实 | 证据 | 源码入口 |
|---|---|---|
| Axum 把哪条 URL 交给 handler？ | `stage=axum_handler` | [`server.rs`](../../sgl-model-gateway/src/server.rs) 的 `build_app()`、`v1_chat_completions()` |
| 这条路径使用哪些共享对象？ | `AppState.context` 的字段 | [`app_context.rs`](../../sgl-model-gateway/src/app_context.rs)；本次只关注 `router_config`、`worker_registry`、`policy_registry`、`client` |
| 为什么是这个 router？ | `stage=router_manager`、`router_mode` | [`router_manager.rs`](../../sgl-model-gateway/src/routers/router_manager.rs) 的 `route_chat()` |
| policy 在什么集合中选择？ | `candidate_workers`、`available_workers`、`policy` | [`http/router.rs`](../../sgl-model-gateway/src/routers/http/router.rs) 的 `select_worker_for_model()` |
| 实际打到了哪里？ | `worker_url`、`route`、`status` | 同文件的 `send_typed_request()` |

### 操作：跑一条可复现的请求

以下命令都从仓库根目录运行，需要 `cargo`、`curl` 和 `rg`。使用 fake worker，不需要 GPU 或
模型权重；它只模拟 HTTP 协议。端口固定为 `19000`（gateway）与 `19001`（worker）。若端口已被
占用，把下面两个端口整体替换为未使用端口。

fake worker 不上报虚假的 `model_path` 或 `tokenizer_path`，因此本实验不会下载 Hugging Face
tokenizer；`/v1/tokenize` 与 `/v1/detokenize` 不在本阶段验证范围内。

先编译一次，避免启动时混入编译日志：

```bash
cargo build --manifest-path sgl-model-gateway/examples/fake-worker/Cargo.toml
cargo build --manifest-path sgl-model-gateway/Cargo.toml --bin smg
```

打开三个终端。

**终端 A：启动一个确定性的 upstream worker。**

```bash
cargo run --manifest-path sgl-model-gateway/examples/fake-worker/Cargo.toml -- \
  --port 19001 --model-id regular-fake
```

**终端 B：以 DEBUG 启动 gateway，并把日志保存到临时文件。**

```bash
cargo run --manifest-path sgl-model-gateway/Cargo.toml --bin smg -- launch \
  --worker-urls http://127.0.0.1:19001 \
  --policy round_robin \
  --host 127.0.0.1 --port 19000 \
  --log-level debug 2>&1 | tee /tmp/smg-stage2.log
```

**终端 C：确认准备就绪，再发一条带固定 request ID 的非流式请求。**

```bash
curl --fail --silent --show-error http://127.0.0.1:19000/readiness

curl --silent --show-error --dump-header - \
  http://127.0.0.1:19000/v1/chat/completions \
  -H 'content-type: application/json' \
  -H 'x-request-id: learn-chat-001' \
  -d '{"model":"regular-fake","messages":[{"role":"user","content":"hello"}]}'
```

期望：HTTP 状态为 `200`，响应头里有 `x-request-id: learn-chat-001`，body 是 fake worker 返回的
chat completion。若 `/readiness` 返回非 200，先查看终端 A 是否仍在运行；不要直接开始读 router。

### 操作：按日志逐层回跳源码

终端 C 执行：

```bash
rg 'learning path|learn-chat-001' /tmp/smg-stage2.log
curl --silent http://127.0.0.1:19001/__fake__/state
```

日志会按下列顺序出现。每读到一行，立刻在对应文件中读到下一次函数调用；不要先展开整个目录。

| 日志 `stage` | 此刻发生什么 | 下一跳与要回答的问题 |
|---|---|---|
| `axum_handler` | `/v1/chat/completions` 被 `v1_chat_completions()` 接住。 | 看 handler 传入的 `headers`、`body`、`body.model`；它不选 worker，只调用 `RouterTrait::route_chat()`。 |
| 无单独日志 | handler 持有 `AppState.context`。 | 打开 `app_context.rs`，只确认 `worker_registry` 提供候选集、`policy_registry` 提供 policy、`client` 负责 upstream HTTP；其余组件留到后续阶段。 |
| `router_manager` | `RouterManager` 已选中一个 router。 | 看 `route_chat()` 中的 `resolve_model_id()` 与 `select_router_for_request()`；本实验应为 `router_mode=regular`。 |
| `worker_selection` | HTTP router 已取得 policy 返回的下标，并得到 worker。 | 看 `get_workers_filtered()`、`is_available()` 与 `policy.select_worker()`；解释两个 worker 数为何相同或不同。 |
| `upstream_send` | 已复制允许转发的 header，准备调用 reqwest。 | 看 `send_typed_request()`；确认 URL 是 `worker_url + route`，而非 client 原始 URL。 |
| `upstream_response_headers` | 已收到 worker 的 HTTP status/header。 | 非流式路径继续 `res.bytes()`；流式路径把 `bytes_stream()` 包为 SSE body。 |

fake worker 的 `stats.last_request.path` 应为 `/v1/chat/completions`，`body.model` 应为
`regular-fake`。这提供了网关日志之外的第二份证据：请求确实到达该 upstream。

### 关键边界：本实验没有按 model 过滤 worker

这条命令没有开启 `--enable-igw`。因此 `select_worker_for_model()` 会把
`effective_model_id` 设为 `None`，候选集实际按 **regular worker + HTTP connection** 过滤；
请求中的 `model` 仍会传给 policy，但不用于 registry 的 worker 过滤。

```text
single-router（本实验）: model -> policy 查找；worker 候选集 = regular + HTTP
IGW（--enable-igw）:    model -> router 选择；worker 候选集 = model + regular + HTTP
```

所以日志里的 `candidate_workers` 不是泛指“同 model 的 worker”。只有启用 IGW 后，才把它解释为
指定 model 下的候选集。这是理解阶段 2 时最容易误读的一层。

### 再跑两个小分支

**流式分支：**只改 `stream` 并换一个 request ID。

```bash
curl --no-buffer --silent --show-error \
  http://127.0.0.1:19000/v1/chat/completions \
  -H 'content-type: application/json' \
  -H 'x-request-id: learn-chat-stream-001' \
  -d '{"model":"regular-fake","messages":[{"role":"user","content":"hello"}],"stream":true}'

rg 'learning path|learn-chat-stream-001' /tmp/smg-stage2.log
```

看到 `[DONE]` 后再回到 `send_typed_request()`：`upstream_response_headers` 只证明上游响应头已经
到达，**不代表 SSE 已结束**；真正的 body 会由 `BreakerTrackedStream` 持续转发。

**上游 500 分支：**不用重启 gateway 即可观察重试时哪些阶段重复。

```bash
curl -X POST http://127.0.0.1:19001/__fake__/config \
  -H 'content-type: application/json' \
  -d '{"failure_status":500}'

curl --silent --show-error --dump-header - \
  http://127.0.0.1:19000/v1/chat/completions \
  -H 'content-type: application/json' \
  -H 'x-request-id: learn-chat-500-001' \
  -d '{"model":"regular-fake","messages":[{"role":"user","content":"hello"}]}'

rg 'learning path|learn-chat-500-001' /tmp/smg-stage2.log

curl -X POST http://127.0.0.1:19001/__fake__/config \
  -H 'content-type: application/json' \
  -d '{"failure_enabled":false}'
```

预期 `axum_handler` 和 `router_manager` 各一次；`worker_selection`、`upstream_send`、
`upstream_response_headers` 可能随 retry 重复。重试次数与间隔由 router 配置决定，不要把日志次数
写死。最后在终端 A、B 按 `Ctrl-C` 停止进程。

完成标志：你能不用 IDE 跳转，画出“Axum handler 到某个 upstream worker URL”的调用方向，并说明：

```text
RouterManager 负责选 router
policy          负责在可用候选 worker 中选下标
HTTP Router     负责构造/发送 upstream HTTP，并处理 JSON 或 SSE response
```

## 阶段 3：读 worker 生命周期，再读路由策略

先学习“可选资源是什么”，再学习“如何选择资源”。

### Worker

阅读顺序：

1. [`core/worker.rs`](../../sgl-model-gateway/src/core/worker.rs)：worker 的元数据、负载、健康状态与
   request guard。
2. [`core/worker_registry.rs`](../../sgl-model-gateway/src/core/worker_registry.rs)：worker 如何查找、
   注册、移除和按模型/类型组织。
3. [`core/steps/worker/local/`](../../sgl-model-gateway/src/core/steps/worker/local/)：新 worker 加入时的
   探测、metadata 发现和构建流程。
4. [`core/circuit_breaker.rs`](../../sgl-model-gateway/src/core/circuit_breaker.rs) 与
   [`core/retry.rs`](../../sgl-model-gateway/src/core/retry.rs)：失败不会无限次继续打向坏 worker 的原因。

### Policy

所有策略实现同一个 [`LoadBalancingPolicy`](../../sgl-model-gateway/src/policies/mod.rs) trait。
按复杂度阅读：

1. `round_robin`：只理解候选集合与轮转状态。
2. `random`、`power_of_two`：引入健康度与负载。
3. `consistent_hashing`：理解 routing key 与会话亲和性。
4. `cache_aware`、`prefix_hash`：最后再把 `my-sglang` 的 radix/KV prefix 知识带进来。

此阶段的关键问题是：**策略不是直接调用某个 worker；它只在 registry 给出的候选 worker 中
作选择，并且应避开不健康或已打开熔断器的 worker。**

## 阶段 4：以测试为主线验证行为

测试通常比实现更快回答“这个模块应该保证什么”。每次只带一个问题阅读：

| 问题 | 先读的测试 |
|---|---|
| 多个 worker 会如何分流？ | [`routing/load_balancing_test.rs`](../../sgl-model-gateway/tests/routing/load_balancing_test.rs) |
| worker 怎样增删改？ | [`routing/worker_management_test.rs`](../../sgl-model-gateway/tests/routing/worker_management_test.rs) |
| worker 失败后为什么不再被选中？ | [`reliability/circuit_breaker_test.rs`](../../sgl-model-gateway/tests/reliability/circuit_breaker_test.rs) |
| SSE 如何保持流式响应？ | [`api/streaming_tests.rs`](../../sgl-model-gateway/tests/api/streaming_tests.rs) |
| PD 如何选择 prefill/decode worker？ | [`routing/pd_routing_test.rs`](../../sgl-model-gateway/tests/routing/pd_routing_test.rs) |

先运行单个测试文件，再回跳到最小实现。不要先跑完整 E2E 测试：它们可能依赖 GPU、模型或
额外服务，适合作为后期系统验证。

## 阶段 5：再进入 PD 与 cache-aware

当普通 HTTP 路由、worker 生命周期和策略接口已经清晰后，才进入 SMG 最有 SGLang 特征的部分。

```text
regular: request -> one regular worker
PD:      request -> prefill worker -> KV/bootstrap metadata -> decode worker -> merged response
```

这时回看 `my-sglang` 中的：

- [Scheduler 与 KV 概览](scheduler-kv-overview.md)：prefill/decode 的不同资源压力；
- [数据结构、所有权与不变量](data-structures.md)：KV page、radix prefix 和 lock；
- [动态流程](dynamic-flows.md)：retract、KV 压力与恢复。

然后读 [`routers/http/pd_types.rs`](../../sgl-model-gateway/src/routers/http/pd_types.rs)、
`routers/http` 下的 PD router，以及上述 PD routing 测试。此时关注的不是单 worker 内部如何
生成 token，而是 router 如何选择一对兼容且健康的 prefill/decode worker，并把请求生命周期
跨机器串起来。

## 最后学习的高级主题

按风险和复杂度递增，而不是按目录字母顺序：

1. observability：Prometheus、request ID、OpenTelemetry；
2. gRPC 和多模型 inference gateway；
3. service discovery 与 Mesh/HA；
4. `/v1/responses`、conversations、MCP；
5. WASM middleware、持久化和 Kubernetes 集成。

这些能力建立在“普通路由 + worker 状态 + policy + failure handling”之上。若前四项还不清楚，
直接阅读 MCP 或 Mesh 会很难判断每一层新增的职责。

## 推荐的第一个练习

在不改变生产行为的前提下，为 `round_robin` 策略添加一个仅在测试中使用的约束：排除具有某个
label 的 worker，并为它写一个路由测试。

这个练习足够小，却会迫使你理解：worker metadata、registry 提供的候选集、policy trait、
健康过滤和 router 层集成。完成后，再尝试比较 `power_of_two` 与 `cache_aware` 在相同 worker
集合中的选择依据。
