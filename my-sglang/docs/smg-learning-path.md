# 从 SGLang 运行时到 SGL Model Gateway：学习路线

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

1. [`server.rs`](../../sgl-model-gateway/src/server.rs)：从 `build_app()` 找到路由注册，再看
   `v1_chat_completions()` handler。
2. [`app_context.rs`](../../sgl-model-gateway/src/app_context.rs)：确认这次请求依赖哪些共享组件。
3. [`routers/router_manager.rs`](../../sgl-model-gateway/src/routers/router_manager.rs)：理解运行时如何
   在 HTTP、gRPC、PD 或 OpenAI router 之间选择实现。
4. [`routers/http/`](../../sgl-model-gateway/src/routers/http/)：跟踪选 worker、转发 body/header 和
   处理 streaming 的路径。

完成标志：你能不用 IDE 跳转，画出“Axum handler 到某个 upstream worker URL”的调用方向，
并说清楚哪一层负责选择、哪一层负责转发。

### 用日志对照这条调用链

以 `DEBUG` 级别启动网关后，发送一个带固定 `x-request-id` 的请求：

```bash
sgl-model-gateway --log-level debug ...
curl ... -H 'x-request-id: learn-chat-001' ...
```

筛选 `smg::learning` 后，同一个 request ID 会按以下顺序出现；日志不会包含 prompt、鉴权头或
响应正文：

```text
axum_handler              Axum handler -> RouterTrait::route_chat
router_manager            RouterManager -> selected router
worker_selection          policy -> selected HTTP worker
upstream_send             HTTP router -> worker request
upstream_response_headers HTTP worker -> gateway response
```

`candidate_workers` 是 registry 按 model/worker type/connection mode 找到的集合；
`available_workers` 是再过滤健康度与熔断状态后的集合。`worker_url` 就是本次请求实际转发的
upstream URL；streaming 请求的最后一条日志表示已收到 upstream response headers，不表示 SSE
流已经结束。

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
