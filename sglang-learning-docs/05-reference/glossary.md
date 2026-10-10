# 术语速查：把输入、状态和计算分开

> 源码基准：learning `2aa70e3eb4`，已合入 origin/main `6fc8d9da32`（2026-10-10）。

| 术语 | 大白话 | 别混成 |
|---|---|---|
| token ID | 词表单位的编号 | UTF-8 byte / KV 槽位 |
| embedding | ID 查到的向量 | KV cache |
| prefill | 为输入上下文计算状态 | 首 token 的文字解码 |
| EXTEND | 向已算前缀追加输入位置的模式 | 一定没有前缀缓存 |
| decode | 根据历史状态继续生成 | 重算整个 prompt |
| chunked prefill | 把输入位置分几轮算 | 按 Transformer 层分割 |
| SPLIT_PREFILL | 按相应路径分段执行层 | 普通 token chunk |
| Req | 一条请求的持续状态 | 一个 batch |
| ScheduleBatch | 本轮调度集合与准备数据 | 固定永久集合 |
| ForwardBatch | 模型执行用的数据包 | HTTP JSON |
| prefix cache | token 前缀对应可复用状态 | 文本语义相似缓存 |
| allocator | 空槽位管理 | 状态字节计算 |
| pool | 实际 tensor 存储 | radix tree |
| lock_ref | 正在用，需保护 | 请求优先级 |
| tombstone | 节点仍在，某组件数据已丢 | 数据有效命中 |
| KVLocPlan | 某轮读写地址视图 | token ID |
| stream | 设备操作队列 | class |
| Graph replay | 重放捕获的执行结构 | 只运行下一层 |
| padding | 为 shape/对齐补容量 | 真正请求边界 |
| TP / DP / EP / PP | 张量/数据/专家/流水线并行 | 同一种多卡方式 |
| PD | prefill/decode 跨实例 | 张量分片 |
| HiCache | 多层状态保存和恢复 | 直接在磁盘上算 Attention |
| speculative | 提候选并批量验证 | 不经验证直接输出 draft |
| TTFT | 客户端看到首 token 的耗时 | 单个 kernel 时间 |
| ITL | 输出 token 间延迟 | 仅 decode 计算时间 |
| SSE | HTTP 流式事件 | 设备 stream |

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/model_executor/forward_batch_info.py](../../python/sglang/srt/model_executor/forward_batch_info.py) | ForwardMode |
| [python/sglang/srt/mem_cache/unified_cache/components/README.md](../../python/sglang/srt/mem_cache/unified_cache/components/README.md) | 统一树术语 |
