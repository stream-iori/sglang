# 当前 Server 启动：先解析，再装配进程

> 源码基准：learning `2aa70e3eb4`，已合入 origin/main `6fc8d9da32`（2026-10-10）。

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

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/launch_server.py](../../python/sglang/launch_server.py) | 入口分支 |
| [python/sglang/srt/entrypoints/engine.py](../../python/sglang/srt/entrypoints/engine.py) | 进程装配 |
| [python/sglang/srt/managers/scheduler.py](../../python/sglang/srt/managers/scheduler.py) | init_model_worker |
| [python/sglang/srt/runtime_context.py](../../python/sglang/srt/runtime_context.py) | 进程配置发布 |
