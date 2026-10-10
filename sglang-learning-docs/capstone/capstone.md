# 结业：用一份可复现报告讲清当前 SRT

> 源码基准：learning `2aa70e3eb4`，已合入 origin/main `6fc8d9da32`（2026-10-10）。

目标是一份别人能照着运行和核对的报告，而不是背下所有函数名。

| 交付 | 必须包含 |
|---|---|
| 请求架构图 | API、编码、调度、执行、解码的进程边界 |
| 状态图 | Req / batch / ForwardBatch、输入输出长度和 KV 生命周期 |
| 当前代码地图 | RuntimeContext、统一缓存、KVLocPlan、runner 分支 |
| 复现命令 | 模型路径、依赖、启动参数、验证脚本 |
| 证据 | server_info、cold/warm、chat/SSE、日志与结果 |
| 范围 | 哪些实测，哪些只读源码，哪些需要 GPU |

## 推荐题目

```text
“Qwen3-0.6B 在 Torch MPS 上的完整请求与缓存复用”
  → 对照两个相同 prompt
  → 解释 198 输入 / 197 cached 的观察
  → 指出默认 UnifiedRadixCache
  → 解释新 token 的采样与下一轮 KV
  → 给出 CUDA Graph/多卡的学习边界
```

如果包含代码修复，补上具体触发、前后行为和必要回归验证。提交前检查 diff 和工作区，不把模型、虚拟环境或临时日志加入 Git。

验收问题：能否解释“缓存树命中但状态不可用”“原始参数不同于有效配置”“固定 shape 但边界数值可变化”？能解释这些，才算理解本次架构改造的核心。

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/managers/scheduler.py](../../python/sglang/srt/managers/scheduler.py) | 请求生命周期 |
| [python/sglang/srt/mem_cache/unified_radix_cache.py](../../python/sglang/srt/mem_cache/unified_radix_cache.py) | 统一 cache |
| [python/sglang/srt/model_executor/model_runner.py](../../python/sglang/srt/model_executor/model_runner.py) | 执行选择 |
