# Rust 前端、processor 和缓存树的边界

> 源码基准：上游 `16eb87c06f`；本地集成 `4f0e1cbea2`（2026-10-20）。本地日志增强与验证范围见总索引和 Mac 验证记录。

**当前独立 `sglang-renderer` 已移除；不要继续沿旧 renderer README 启动或推断架构。** Rust HTTP 前端和请求处理主线从 `sglang-server` 进入。

```text
Python HTTP → TokenizerManager ───────────────┐
                                             ↓
Rust HTTP → server / processor → typed IPC → Python Scheduler → Torch 执行
                                             ↑
UnifiedRadixCache → tree_core_registry → Python / Rust TreeCore
```

图表示不同入口的职责，普通 Mac 脚本仍运行 Python HTTP 路径。

| 组件 | 当前职责 | 不能据此推断 |
|---|---|---|
| sglang-server | Rust HTTP 前端、请求处理、与 Python scheduler 的类型化边界 | Python 模型执行已被全部替换 |
| sglang-processor | template、tokenizer、reasoning/tool 等处理 | 连续批次调度在这里完成 |
| sglang-radix-tree | 统一树的 Rust 实现 | 所有 KV 张量都由 Rust 独立管理 |
| sgl-model-gateway | 跨 engine/provider 路由与连接 | 等同于某个 engine 的 Scheduler |

## 阅读协议边界

server README 要求类型化边界和单一协议 schema。先看外部请求如何变成归一化数据，再看与 Python 消息的映射，不能只比较 JSON 字段名字是否相同。

| 核对项 | 失败后果 |
|---|---|
| 模型、tokenizer 与 revision | 相同 ID 在两端含义可能不同 |
| chat template、thinking | 相同 messages 产生不同 prompt |
| 长度、停止、reasoning/tools | 丢 token、提前结束或响应结构错误 |
| 累计/增量 SSE 语义 | 重复拼接或漏文本 |
| 请求关联、abort、flush 与 server_info | 控制请求与实际 engine 状态不一致 |

本次上游已有 Rust 前端 flush_cache、启动时 server_info 字段和 typed API 等更新，需沿各处理器读到 Python 对端；不能用 Python 接口成功替代 Rust 端到端验证。

## 练习与验证边界

在两种前端上给定同一模型与 messages，比对渲染 prompt、输入 ID、停止原因和流式拼接文本。验证顺序是 processor 的 prompt/token parity，再做真实服务协议。

本次未构建或启动 Rust 服务；本地 MPS 结果只覆盖 Python HTTP。使用 Rust 路径时按当前 crate 的构建与启动说明另行验证。

## 对照源码

| 文件 | 用途 |
|---|---|
| [server README](../../rust/sglang-server/README.md) | 当前职责与协议原则 |
| [server src](../../rust/sglang-server/src) | HTTP 与处理流程 |
| [processor src](../../rust/sglang-processor/src) | 渲染、分词与解析 |
| [radix-tree src](../../rust/sglang-radix-tree/src) | Rust 树实现 |
| [tree_core_registry](../../python/sglang/srt/mem_cache/unified_cache/tree_core_registry.py) | Python/Rust 树选择 |
