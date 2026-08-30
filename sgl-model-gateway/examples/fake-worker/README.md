# SGL Model Gateway Fake Worker

这是一个本地、确定性的 SGLang HTTP worker 协议模拟器。它用于调试 SGL Model Gateway 的
worker discovery、路由策略、SSE、重试、熔断和 PD dispatch；它不加载模型，也不实现真实 KV
传输。

```bash
cargo run --manifest-path sgl-model-gateway/examples/fake-worker/Cargo.toml -- --port 8001
```

默认仅绑定 `127.0.0.1`。常用控制接口：

```bash
curl http://127.0.0.1:8001/__fake__/state
curl -X POST http://127.0.0.1:8001/__fake__/config \
  -H 'content-type: application/json' \
  -d '{"delay_ms":200,"failure_status":500}'
curl -X POST http://127.0.0.1:8001/__fake__/config \
  -H 'content-type: application/json' \
  -d '{"failure_enabled":false,"health":"unhealthy"}'
```

运行常规和 PD smoke check：

```bash
bash sgl-model-gateway/examples/fake-worker/scripts/smoke.sh
```
