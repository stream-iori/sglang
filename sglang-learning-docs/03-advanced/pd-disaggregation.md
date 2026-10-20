# PD 学习与验证顺序

> 源码基准：上游 `16eb87c06f`；本地集成 `4f0e1cbea2`（2026-10-20）。本地日志增强与验证范围见总索引和 Mac 验证记录。

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

## 实验前先生成配置清单

```bash
rg -n 'disaggregation_mode|disaggregation_transfer_backend|bootstrap|retraction' python/sglang/srt/arg_groups/fields/disagg.py
rg --files test/registered | rg 'disaggregation|pd_'
```

按当前测试对应的模型与后端准备匹配硬件，不复制未知机器的 IP、网卡或环境变量。实验记录双方源码提交、模型 revision、dtype、布局、TP 和传输库版本。

| 顺序 | 对照组 | 验收 |
|---|---|---|
| 1 | 同模型单实例 | 正确输出与停止原因 |
| 2 | PD、低并发 | 接收前分配、传输完成后运行 |
| 3 | 长 prompt / 短输出 | 区分 prefill、传输、decode 时间 |
| 4 | 短 prompt / 长输出 | 观察 decode 负载与传输成本 |
| 5 | 并发、abort、容量紧张 | 队列能推进，资源能安全回收 |

答案不能只写“PD 更快”：需要说明哪类负载获益、哪个阶段变慢及对应证据。

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/arg_groups/fields/disagg.py](../../python/sglang/srt/arg_groups/fields/disagg.py) | 当前参数 |
| [python/sglang/srt/disaggregation/decode.py](../../python/sglang/srt/disaggregation/decode.py) | 队列 |
| [python/sglang/srt/disaggregation/mooncake/conn.py](../../python/sglang/srt/disaggregation/mooncake/conn.py) | 一种传输实现 |
| [python/sglang/srt/disaggregation/nixl/conn.py](../../python/sglang/srt/disaggregation/nixl/conn.py) | 另一种传输实现 |
