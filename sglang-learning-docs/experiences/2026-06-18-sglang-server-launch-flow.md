# 当前 Server 启动：先解析，再装配进程

> 源码基准：上游 `16eb87c06f`；本地集成 `4f0e1cbea2`（2026-10-20）。本地日志增强与验证范围见总索引和 Mac 验证记录。

```text
sglang.launch_server
  → 插件与参数
  → resolve_once / resolving_view
  → 选择 encoder / grpc / Ray / HTTP 路径
  → Engine 进程装配
  → scheduler：权重 / memory pools / Attention / Graph / sampling 预热
  → HTTP 可用
```

`python -m sglang.launch_server` 仍是兼容入口，本地脚本已通过此入口验证。不要把旧入口内部固定行号写成当前事实。

当前模型初始化包括 post-capture 的池调整/预热等条件分支，并非固定“权重读完立刻 ready”。服务是否可用用 health 和生成请求确认。

RuntimeContext 在相应进程发布有效配置；子进程各有自己的状态。新 Rust 服务路径也需按所选配置单独确认。

## 启动日志的验收顺序

```text
进程开始 → 参数解析 → 模型加载 → 池与 backend → 预热 → ready → 真实请求
```

看到监听端口只证明进程在监听，看到权重加载只证明加载阶段完成。health 可能在预热期间返回非 200，验证脚本需要等待并设置超时。

练习：模型加载结束后立刻失败，首先检查哪一段？答案：首个 traceback 和池/Attention/预热阶段，不能跳到“HTTP 客户端有问题”。对每个异常记录实际进程和配置，避免 attach 错进程。

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/launch_server.py](../../python/sglang/launch_server.py) | 入口分支 |
| [python/sglang/srt/entrypoints/engine.py](../../python/sglang/srt/entrypoints/engine.py) | 进程装配 |
| [python/sglang/srt/managers/scheduler.py](../../python/sglang/srt/managers/scheduler.py) | init_model_worker |
| [python/sglang/srt/runtime_context.py](../../python/sglang/srt/runtime_context.py) | 进程配置发布 |
