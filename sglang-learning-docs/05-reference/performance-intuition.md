# 性能直觉：KV 字节、算力、带宽与提交成本

> 源码基准：learning `2aa70e3eb4`，已合入 origin/main `6fc8d9da32`（2026-10-10）。

先估算资源规模，再看 profile。吞吐变化不能只用“GPU 利用率高了”解释。

## 每 token KV 字节

对各层 KV 布局相同的普通 MHA/GQA：

```text
bytes_per_token = 2 × L × Hkv × Dhead × dtype_bytes
                  │   │    │      │          │
                 K/V 层数 KV头数 每头维度 每元素字节
```

| 因子 | 为什么乘 |
|---|---|
| 2 | 每层既存 K，又存 V |
| L | 每个 Attention 层都需要自己的状态 |
| Hkv | 每个位置有若干 KV heads，不是 Q heads |
| Dhead | 一个 head 的 K 或 V 向量包含这么多元素 |
| dtype_bytes | BF16/FP16 通常每元素 2 byte，FP32 为 4 |

### 本地 Qwen3-0.6B 示例

从本地 config.json 读取：L=28，Hkv=8，Dhead=128；本地日志显示 BF16 KV。

```text
一层、一个 token 的 K：8 × 128 × 2 = 2048 byte
一层 K+V：            4096 byte
全部 28 层：          114688 byte = 112 KiB / token
4096 个逻辑 token：   448 MiB（纯 K/V 数据估算）
```

不是 token 文本占 112 KiB；这是模型为该输入位置保留的多层状态。4096 指逻辑容量总和，不是每请求都拥有 4096，且池实现可能有保留槽、分页、对齐等开销。

| 公式不直接覆盖 | 原因 |
|---|---|
| MLA / 压缩 KV | 存储表示不同 |
| SWA 混合层 | 保留长度随层/窗口不同 |
| Mamba/SSM | 存 recurrent state，不是普通 K/V 序列 |
| 量化 KV | 可能增加 scale、特殊 packing/alignment |
| TP 每卡内存 | 分片与复制规则要从实现确认 |
| 总设备内存 | 还包括权重、激活、Graph、临时 workspace |

更一般地，应按各层实际存储形状和 dtype 求和。

## 三类成本

```text
一轮总耗时 ≈ CPU 调度/提交 + 设备计算/内存访问 + 通信/同步
```

| 负载 | 常见观察方向 | 不可直接断言 |
|---|---|---|
| 长 prefill | GEMM/Attention 计算、有效 token 数 | 一定纯算力受限 |
| 小 batch decode | 权重/KV 读取、launch 频率 | 一定只受 KV 带宽限制 |
| 大 batch decode | 算子形状、KV、显存、通信 | batch 越大越快 |
| Graph | CPU launch、padding、capture 内存 | 重放消除全部 CPU 成本 |
| prefix hit | 未命中 token、恢复成本 | 命中高就没有延迟 |

本地 MPS 是统一内存体系，不能把 CUDA HBM、PCIe 和 NCCL 的具体数字当成本机测量。

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/mem_cache/memory_pool.py](../../python/sglang/srt/mem_cache/memory_pool.py) | 实际 KV 形状 |
| [python/sglang/srt/mem_cache/unified_memory_pool.py](../../python/sglang/srt/mem_cache/unified_memory_pool.py) | 子池布局与对齐 |
| [python/sglang/srt/mem_cache/allocation_sizing.py](../../python/sglang/srt/mem_cache/allocation_sizing.py) | 容量计算 |
| [python/sglang/srt/model_executor/model_runner.py](../../python/sglang/srt/model_executor/model_runner.py) | 执行与采样 |
