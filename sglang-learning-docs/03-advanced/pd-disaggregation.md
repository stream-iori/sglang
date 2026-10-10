# PD 学习与验证顺序

> 源码基准：learning `2aa70e3eb4`，已合入 origin/main `6fc8d9da32`（2026-10-10）。

先验证一个 engine，再做两个 engine 的角色配置和状态传输。当前 Mac 学习脚本只启动普通单实例。

| 阶段 | 验证内容 | 失败时先看 |
|---|---|---|
| 单实例 | 同模型普通生成稳定 | 模型、backend、依赖 |
| 角色启动 | prefill/decode 参数与端口符合预期 | 解析后配置、启动日志 |
| 连接 | bootstrap、地址、backend 初始化 | 网络、库、注册信息 |
| 状态接收 | prealloc → transfer → ready | 容量、poll、错误/超时 |
| 输出一致 | prompt、token IDs、停止与流式 | template、metadata、输出路径 |
| 性能 | TTFT/ITL、transfer、队列 | 负载与传输开销 |

```text
路由层 → prefill worker
         │ KV 与 metadata
         ▼
       decode worker → 生成结果 → 客户端
```

部署参数先查当前 `--help` 和 `arg_groups/fields/disagg.py`。远端地址、RDMA 网卡、传输库配置依赖机器环境，不能把一台机器的网卡名照搬到另一台。

长输入、长输出、并发和冷热缓存要分开测。PD 的目标是隔离资源和分阶段扩容，不保证每个请求的端到端延迟更小。

源码细节见 [PD 队列与传输](disaggregation-source.md)。本次没有执行 PD、多机或 RDMA 测试。

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/arg_groups/fields/disagg.py](../../python/sglang/srt/arg_groups/fields/disagg.py) | 当前参数 |
| [python/sglang/srt/disaggregation/decode.py](../../python/sglang/srt/disaggregation/decode.py) | 队列 |
| [python/sglang/srt/disaggregation/mooncake/conn.py](../../python/sglang/srt/disaggregation/mooncake/conn.py) | 一种传输实现 |
| [python/sglang/srt/disaggregation/nixl/conn.py](../../python/sglang/srt/disaggregation/nixl/conn.py) | 另一种传输实现 |
