# Model Gateway：选择 engine 的层

> 源码基准：上游 `16eb87c06f`；本地集成 `4f0e1cbea2`（2026-10-20）。本地日志增强与验证范围见总索引和 Mac 验证记录。

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

## 路由与缓存热度的因果链

```text
相同前缀稳定路由到同一 engine → 更可能命中该 engine 的缓存
过度集中 → 排队增加 → TTFT 可能变差
```

因此缓存亲和性、负载均衡与失败重试要一起观察。重试前还要考虑是否已向客户端发送流式片段，不能无条件把同一输出再拼一次。

## 无模型与真实模型两种验收

| 实验 | 能证明 | 不能证明 |
|---|---|---|
| fake-worker | 请求转发、路由策略、错误处理、协议拼接 | 模型正确、KV 命中或 GPU 性能 |
| 单真实 engine | Gateway 到模型的完整协议 | 多 worker 负载均衡 |
| 多真实 engine | 路由、热度、排队和失败场景 | 其他模型或其他后端同样表现 |

本轮只更新源码解释和验证步骤，不把 fake-worker 的返回当作真实模型输出。本地直接访问 MPS 服务是最短学习主线。

## 对照源码

| 文件 | 读什么 |
|---|---|
| [sgl-model-gateway/src/main.rs](../../sgl-model-gateway/src/main.rs) | 启动入口 |
| [sgl-model-gateway/src/routers](../../sgl-model-gateway/src/routers) | 路由实现 |
| [sgl-model-gateway/examples/fake-worker/README.md](../../sgl-model-gateway/examples/fake-worker/README.md) | 无模型验证入口 |
