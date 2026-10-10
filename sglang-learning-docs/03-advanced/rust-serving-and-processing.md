# Rust 组件：服务、处理、树和网关分别是什么

> 源码基准：learning `2aa70e3eb4`，已合入 origin/main `6fc8d9da32`（2026-10-10）。

当前仓库有多项 Rust 能力，职责不同。看到 Rust 代码增加，不代表 Python Scheduler 或 Torch ModelRunner 已经被整体替换。

| 组件 | 负责什么 | 不应混同 |
|---|---|---|
| sglang-server | Rust 原生服务/API 入口 | 模型 GPU kernel |
| sglang-processor | chat 渲染、分词、reasoning/tools 等处理 | 请求批次调度 |
| sglang-renderer | 独立 OpenAI 前端，借 processor 并向 engine 提交 IDs | 普通 Mac 启动脚本默认服务 |
| sglang-radix-tree | Unified TreeCore 的 Rust 实现 | KV 字节全部改由 Rust 存储 |
| sgl-model-gateway | 路由到多个 engine/provider，管理请求协议等 | 单 worker 的 Attention |

## Python 主线与 Rust 前端的边界

```text
Python HTTP 前端 → TokenizerManager → Scheduler → Torch 执行

Rust renderer → processor → engine /generate → 引擎执行
                    │
             template / token IDs / 解析

UnifiedRadixCache → tree_core_registry → Python 或 Rust TreeCore
```

图表达职责，实际启动/IPC 依赖所选服务路径。Mac 本地验证仍是 Python HTTP 主线，未切到 Rust 服务。

## 需要核对的一致性

| 项目 | 为什么 |
|---|---|
| model/revision/tokenizer | 前端和 engine 的 ID 语义要一致 |
| chat template / thinking 开关 | 同一 messages 应生成一致输入 |
| reasoning/tool parser | 输出结构不是只 decode 一个字符串 |
| cumulative / incremental stream | 防止重复拼接或漏片段 |
| stop/min tokens | 前后端停止边界要一致 |

`sglang-renderer/README.md` 明确把当前独立 renderer 标为临时方案，后续职责可能转移。学习文档应保留这个状态，而不是把它写成永久唯一架构。

## 对照源码

| 文件 | 读什么 |
|---|---|
| [rust/sglang-renderer/README.md](../../rust/sglang-renderer/README.md) | 职责与临时状态 |
| [rust/sglang-processor/src](../../rust/sglang-processor/src) | 渲染、分词和 parser |
| [rust/sglang-server/src](../../rust/sglang-server/src) | 服务入口 |
| [rust/sglang-radix-tree/src](../../rust/sglang-radix-tree/src) | TreeCore |
| [python/sglang/srt/mem_cache/unified_cache/tree_core_registry.py](../../python/sglang/srt/mem_cache/unified_cache/tree_core_registry.py) | 树实现选择 |
