# 用 Fake Worker 本地调试 SGL Model Gateway

这一节把 [SMG 学习路线](smg-learning-path.md) 中的 HTTP 路由、worker lifecycle 和 PD
概念变成一个不需要 GPU 或模型权重的本地实验。实验使用
[`sgl-model-gateway/examples/fake-worker`](../../sgl-model-gateway/examples/fake-worker)：它模拟
SGLang worker 的 HTTP 协议，不模拟模型推理。

## 它模拟什么，不模拟什么

Fake worker 可以回答“SMG 如何把请求路由到一个上游服务”：

```text
SMG -> /health, /server_info, /model_info, /v1/models
SMG -> /generate 或 /v1/chat/completions -> JSON / SSE response
```

它还可以在运行时模拟健康变化、延迟、5xx 和 SSE 中断。因此适合学习 worker discovery、
负载均衡、retry、circuit breaker 和 PD 的成对 dispatch。

它不加载模型、不运行 `my-sglang` 的 scheduler、不持有真实 KV cache，也不建立 bootstrap
server。因此它不能验证模型质量、KV transfer 或真实 prefill/decode 性能；这些问题仍需真实
SGLang worker。

## 前置条件与目录约定

以下命令从仓库根目录运行。Fake worker 是独立 example crate，不修改 SMG 根 crate 或现有
测试 MockWorker；这样在同步上游时不会与 `tests/common/mock_worker.rs` 的高频改动冲突。

```bash
fake_manifest=sgl-model-gateway/examples/fake-worker/Cargo.toml
gateway_manifest=sgl-model-gateway/Cargo.toml
```

每个 fake worker 默认只监听 `127.0.0.1`。它的本地控制接口是 `__fake__` 前缀，不属于
SGLang API，也不应暴露到生产网络。

## 实验一：普通 HTTP 路由

打开三个终端。

终端 A 启动一个上游 worker：

```bash
cargo run --manifest-path "$fake_manifest" -- --port 18001 --model-id regular-fake
```

终端 B 启动 SMG：

```bash
cargo run --manifest-path "$gateway_manifest" --bin smg -- launch \
  --worker-urls http://127.0.0.1:18001 \
  --policy round_robin \
  --host 127.0.0.1 --port 18000
```

终端 C 通过 SMG 发请求：

```bash
curl http://127.0.0.1:18000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"regular-fake","messages":[{"role":"user","content":"hello"}]}'
```

再请求流式响应：

```bash
curl --no-buffer http://127.0.0.1:18000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"regular-fake","messages":[{"role":"user","content":"hello"}],"stream":true}'
```

此时查看 fake worker 记录到的上游请求：

```bash
curl http://127.0.0.1:18001/__fake__/state
```

这条链路把 [SMG 学习路线](smg-learning-path.md) 的最小模型具体化：SMG 先发现/注册 worker，
再由 router 将请求转发到 worker，最后把 JSON 或 SSE 返回给 client。

## 实验二：在不中断 SMG 的情况下制造故障

让 worker 每个请求延迟 200 ms 并返回 500：

```bash
curl -X POST http://127.0.0.1:18001/__fake__/config \
  -H 'content-type: application/json' \
  -d '{"delay_ms":200,"failure_status":500}'
```

再次经 SMG 发请求，观察 gateway 的错误、retry 和 circuit breaker 日志。恢复正常：

```bash
curl -X POST http://127.0.0.1:18001/__fake__/config \
  -H 'content-type: application/json' \
  -d '{"failure_enabled":false,"delay_ms":0}'
```

也可以直接模拟健康检查失败：

```bash
curl -X POST http://127.0.0.1:18001/__fake__/config \
  -H 'content-type: application/json' \
  -d '{"health":"unhealthy"}'
```

此操作只影响 fake worker 的 `/health` 与 `/health_generate` 响应；它可以帮助区分
“请求处理失败”与“worker 已不应加入候选集”两种 SMG 行为。

## 实验三：PD 路由形状

PD 的核心不是让一个 fake worker 真的传 KV，而是让 SMG 选中一对 prefill/decode worker，
并向两端发出带 bootstrap 字段的请求：

```text
client request
   -> SMG selects (prefill, decode)
   -> prefill receives bootstrap_host/bootstrap_port/bootstrap_room
   -> decode receives paired request
   -> SMG returns decode response
```

终端 A 与 B 分别启动 prefill 和 decode fake worker：

```bash
cargo run --manifest-path "$fake_manifest" -- --port 18011 --model-id pd-fake --worker-type prefill
cargo run --manifest-path "$fake_manifest" -- --port 18012 --model-id pd-fake --worker-type decode
```

终端 C 启动 PD gateway：

```bash
cargo run --manifest-path "$gateway_manifest" --bin smg -- launch \
  --pd-disaggregation \
  --prefill http://127.0.0.1:18011 19011 \
  --decode http://127.0.0.1:18012 \
  --policy round_robin \
  --host 127.0.0.1 --port 18010
```

终端 D 发送 `/generate`，随后分别查看两个 fake worker 的状态：

```bash
curl http://127.0.0.1:18010/generate \
  -H 'content-type: application/json' \
  -d '{"text":"pd request","stream":false}'

curl http://127.0.0.1:18011/__fake__/state
curl http://127.0.0.1:18012/__fake__/state
```

prefill 的 `last_request.body` 中应能看到 `bootstrap_host`、`bootstrap_port` 和
`bootstrap_room`。这是 SMG PD router 的协议加工；fake worker 不会消费这些值建立真实 KV
通道。

## 自动 smoke check

以下脚本依次验证普通 JSON、SSE 和 PD 请求形状：

```bash
bash sgl-model-gateway/examples/fake-worker/scripts/smoke.sh
```

如果要继续深入，先回到 [SMG 学习路线](smg-learning-path.md) 的 worker registry 和 policy
阶段，再用多个 fake worker 对比 `round_robin`、`power_of_two` 与故障恢复的选择结果。
