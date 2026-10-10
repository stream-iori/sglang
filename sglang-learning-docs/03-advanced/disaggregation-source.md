# PD 源码：先拿到容量，再接收 KV，再运行

> 源码基准：learning `2aa70e3eb4`，已合入 origin/main `6fc8d9da32`（2026-10-10）。

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

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/disaggregation/prefill.py](../../python/sglang/srt/disaggregation/prefill.py) | prefill sender |
| [python/sglang/srt/disaggregation/decode.py](../../python/sglang/srt/disaggregation/decode.py) | prealloc/transfer/运行队列 |
| [python/sglang/srt/disaggregation/base/conn.py](../../python/sglang/srt/disaggregation/base/conn.py) | 连接契约 |
| [python/sglang/srt/disaggregation/common/staging_handler.py](../../python/sglang/srt/disaggregation/common/staging_handler.py) | staging 生命周期 |
| [python/sglang/srt/disaggregation/decode_hicache_mixin.py](../../python/sglang/srt/disaggregation/decode_hicache_mixin.py) | cache 恢复衔接 |
