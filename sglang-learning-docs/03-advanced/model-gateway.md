# Model Gateway：选择 engine 的层

> 源码基准：learning `2aa70e3eb4`，已合入 origin/main `6fc8d9da32`（2026-10-10）。

Gateway 在引擎外部决定请求去哪个 worker/provider，处理协议、路由和连接等工作。它不代替 worker 内部的连续批次调度。

```text
客户端 → Gateway
             ├─ engine A → 自己的 Scheduler / cache / 模型
             ├─ engine B → 自己的 Scheduler / cache / 模型
             └─ provider / 其他路由路径
```

| 决策层 | 例子 |
|---|---|
| Gateway | 同一个 prompt 去哪个 engine；失败是否重试 |
| Scheduler | 到达此 engine 的请求谁先参与本轮 |
| cache | 这个 engine 是否已有可复用前缀 |
| Runner | 本轮用 eager 还是可重放 Graph |

路由选择会影响前缀缓存热度，但它不意味着两个 engine 自动共享 KV。PD 路由还有角色匹配和传输契约。

## 学习验证顺序

1. 用 fake-worker 理解请求转发和协议，不加载真实模型。
2. 查询当前 gateway 二进制 `--help`，按所选路由准备配置。
3. 接入单个真实 engine，验证 health、非流式和 SSE。
4. 多 worker 下检查路由、重试、缓存热度与负载。

本次文档改造未跑完整 gateway 冒烟，不把曾经的编译/帮助输出写成服务验证通过。

## 对照源码

| 文件 | 读什么 |
|---|---|
| [sgl-model-gateway/src/main.rs](../../sgl-model-gateway/src/main.rs) | 启动入口 |
| [sgl-model-gateway/src/routers](../../sgl-model-gateway/src/routers) | 路由实现 |
| [sgl-model-gateway/examples/fake-worker/README.md](../../sgl-model-gateway/examples/fake-worker/README.md) | 无模型验证入口 |
