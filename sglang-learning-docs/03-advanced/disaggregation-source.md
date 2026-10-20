# PD 源码：先拿到容量，再接收 KV，再运行

> 源码基准：上游 `16eb87c06f`；本地集成 `4f0e1cbea2`（2026-10-20）。本地日志增强与验证范围见总索引和 Mac 验证记录。

PD 把同一请求的 prefill 与 decode 放到不同 engine。跨实例传的是状态和 metadata，不能只把首个输出 token 发过去就继续。

```text
Prefill engine                         Decode engine
计算 prompt → KV / metadata         请求进入 prealloc queue
          │                          预算与接收槽准备
          └──────── transfer ─────→  transfer queue
                                     检查完成 / 错误
                                          ↓
                                  PREBUILT / 可运行 batch
                                          ↓
                                       decode
```

## 当前主要对象

| 对象/模块 | 问题 |
|---|---|
| DecodePreallocQueue | 接收前能否保证容量；没有容量怎么办 |
| DecodeTransferQueue | 哪些请求仍在传输、哪些已经就绪 |
| process_decode_queue | 队列之间怎样移动、回退请求怎样恢复 |
| base/conn.py + backend conn | sender/receiver、poll、连接和传输契约 |
| common/staging_* | staging 缓冲及其资源生命周期 |
| decode_hicache_mixin.py | PD 与缓存/host 恢复如何衔接 |

## 不能省略的状态

| 状态 | 省略后的风险 |
|---|---|
| bootstrap / 请求关联信息 | 对不到正确实例或请求 |
| KV 布局/槽位映射 | 接收到错误地址或不兼容布局 |
| metadata / 长度 | decode 读错上下文范围 |
| 传输完成信号 | 数据未就绪就被读取 |
| 中止/超时/回退生命周期 | 泄漏槽位、重复释放、请求永久等待 |

PREBUILT 只表达特定的就绪批次模式，不能理解成“已经生成所有 token”。

不同传输 backend 的能力不一样。先确认当前配置选 Mooncake、NIXL 或其他实现，再读该 backend 的连接逻辑。

## 请求完成之前，传输资源也要完成

当前 PD decode 的 idle 判断还需考虑延后 KV 释放等未完成工作。用户侧输出结束，不代表 staging、传输句柄和槽位可以立即复用。

```text
排队 → 预分配 → 传输中 → 就绪 → decode → 输出结束
                    └─ 超时/中止 ───────────┐
                                          ↓
                         等待安全边界 → 回收各级资源
```

| 故障注入场景 | 应有结果 |
|---|---|
| Decode 暂时无容量 | 等待或按配置处理，不向未分配位置写入 |
| 传输过程中 abort | 控制面结束请求，数据面仍按安全边界清理 |
| backend 报错 | 有明确错误与资源收尾，不无限留在 transfer queue |
| PD 角色切换 | 核对当前 Graph/池的兼容与复用条件 |

当前上游包含 PD 角色切换时复用启动 decode Graph 的修复。它不意味着所有角色切换都不需要检查配置、池或传输状态。

## 验收材料

同一 rid 至少记录：在哪个实例 prefill、接收端预留容量、传输完成、首轮 decode、结束与清理。若只有两个实例都 health 200，只能证明它们启动，不能证明 PD 状态链已跑通。

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/disaggregation/prefill.py](../../python/sglang/srt/disaggregation/prefill.py) | prefill sender |
| [python/sglang/srt/disaggregation/decode.py](../../python/sglang/srt/disaggregation/decode.py) | prealloc/transfer/运行队列 |
| [python/sglang/srt/disaggregation/base/conn.py](../../python/sglang/srt/disaggregation/base/conn.py) | 连接契约 |
| [python/sglang/srt/disaggregation/common/staging_handler.py](../../python/sglang/srt/disaggregation/common/staging_handler.py) | staging 生命周期 |
| [python/sglang/srt/disaggregation/decode_hicache_mixin.py](../../python/sglang/srt/disaggregation/decode_hicache_mixin.py) | cache 恢复衔接 |
