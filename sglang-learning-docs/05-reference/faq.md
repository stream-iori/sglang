# FAQ: 常见概念混淆澄清

> 初学 SGLang 源码时最容易搞混的概念，一次性解答清楚。
> 建议在学习 [Week 1](../01-architecture/foundations.md)-[Week 2](../02-core-systems/scheduler-and-cache.md) 时遇到困惑随时查阅。
>
> **关联文档**:
> - Q1-Q2 (数据结构) 配合 [model-execution.md: 数据流](../02-core-systems/model-execution.md)
> - Q3 (RadixCache vs KVPool) 配合 [scheduler-and-cache.md: 内存池](../02-core-systems/scheduler-and-cache.md) 和 [06-demo 实验 2](../04-practice/exercises.md)
> - Q4 (event_loop) 配合 [foundations.md: Scheduler 架构](../01-architecture/foundations.md)
> - Q5-Q6 (ForwardMode / Chunked Prefill) 配合 [performance-intuition.md](../05-reference/performance-intuition.md)

---

## Q1: ScheduleBatch vs ForwardBatch vs ModelWorkerBatch — 为什么要三种？

这三者是同一批请求在不同阶段的"形态"。

```
ScheduleBatch (CPU, Scheduler 内部)
    │
    │  get_model_worker_batch()  — 提取可序列化的数据
    ▼
ModelWorkerBatch (CPU, 可跨进程传输)
    │
    │  ForwardBatch.init_new()  — 转为 GPU Tensor
    ▼
ForwardBatch (GPU, ModelRunner 使用)
```

| 类型 | 在哪里 | 包含什么 | 为什么需要 |
|---|---|---|---|
| **ScheduleBatch** | Scheduler 进程 (CPU) | `List[Req]`, forward_mode, tree_cache 引用 | Scheduler 用来做调度决策，需要完整的请求对象 |
| **ModelWorkerBatch** | 跨进程传输 | 纯 Python 数据 (list, int) | Req 对象不可序列化，需要提取关键字段传给 Worker |
| **ForwardBatch** | ModelRunner (GPU) | `torch.Tensor` (input_ids, seq_lens, ...) | 模型 forward 只接受 Tensor，不认识 Python 对象 |

**类比**: 就像寄快递:
- ScheduleBatch = 你手里的一堆东西 (原始形态)
- ModelWorkerBatch = 装进快递盒并贴好标签 (可运输形态)
- ForwardBatch = 收件人拆出来放到自己桌上 (使用形态)

**源码位置**:
- ScheduleBatch: `python/sglang/srt/managers/schedule_batch.py`
- ForwardBatch: `python/sglang/srt/model_executor/forward_batch_info.py`
- 转换: `schedule_batch.py` 中的 `get_model_worker_batch()`

---

## Q2: Req vs GenerateReqInput vs TokenizedGenerateReqInput — 请求的生命周期

一个请求经历三种形态，逐步丰富信息：

```
用户 HTTP 请求 (JSON)
    │
    │  FastAPI 解析
    ▼
GenerateReqInput  ← "用户说了什么" (原始文本/参数)
    │
    │  TokenizerManager: tokenize + chat template
    ▼
TokenizedGenerateReqInput  ← "准备好送给 Scheduler" (token IDs)
    │
    │  Scheduler: process_input_requests()
    ▼
Req  ← "正在被处理的请求" (内部状态: output_ids, KV 位置, finished 等)
```

| 类型 | 创建位置 | 包含 | 不包含 |
|---|---|---|---|
| **GenerateReqInput** | HTTP Server | text, sampling_params, images | token_ids (还没 tokenize) |
| **TokenizedGenerateReqInput** | TokenizerManager | input_ids, sampling_params | output_ids, KV 位置 (还没调度) |
| **Req** | Scheduler | 一切: input_ids, output_ids, origin_input_ids, req_pool_idx, finished... | — |

**为什么不直接用一个类？**

1. **职责分离**: HTTP 层不应该知道 token 是什么；Scheduler 不应该做 tokenize
2. **进程隔离**: 每种形态在不同进程中，跨进程传递需要轻量对象
3. **信息递增**: 后一种包含前一种的信息 + 新增的处理结果

**源码位置**: `python/sglang/srt/managers/io_struct.py`

---

## Q3: RadixCache vs TokenToKVPool — 都叫"缓存"但完全不同

| | RadixCache | TokenToKVPool |
|---|---|---|
| **是什么** | 前缀索引树 (哪些 token 序列被缓存了) | GPU 显存池 (KV 数据实际存在哪里) |
| **数据结构** | Radix Tree (树节点) | 连续显存块 + 分配器 |
| **存在哪** | CPU 内存 | GPU 显存 |
| **解决什么问题** | "这个 token 序列以前见过吗？" | "KV 数据放在 GPU 哪个地址？" |
| **类比** | 图书馆目录卡 (告诉你书在哪个架子) | 书架本身 (实际存放书籍) |

**它们如何协作**:

```
新请求到达: tokens = [1, 2, 3, 4, 5]

Step 1: RadixCache.match_prefix([1,2,3,4,5])
        → 命中 [1,2,3]，返回对应的 KV 位置 [page_10, page_11, page_12]
        → 这些位置指向 TokenToKVPool 中的实际 GPU 数据

Step 2: 只需要为 [4, 5] 分配新 KV 空间
        → TokenToKVPool.alloc(2) → [page_20, page_21]

Step 3: 只对 [4, 5] 做 prefill，复用 [1,2,3] 的已有 KV

Step 4: 请求完成后
        → RadixCache 保留树节点 (下次可能还有相同前缀)
        → TokenToKVPool 中的页标记为"可被淘汰" (lock_ref -= 1)
```

---

## Q4: event_loop_normal vs event_loop_overlap — 什么时候用哪个

```python
# scheduler.py:1425
def event_loop_normal(self):
    """串行模式: 调度 → forward → 处理结果 → 下一轮"""
    while True:
        recv_reqs = self.request_receiver.recv_requests()
        self.process_input_requests(recv_reqs)
        batch = self.get_next_batch_to_run()
        if batch:
            result = self.run_batch(batch)          # 等 GPU 完成
            self.process_batch_result(batch, result) # 然后处理
        else:
            self.on_idle()

# scheduler.py:1452
def event_loop_overlap(self):
    """重叠模式: 上一轮的 GPU forward 与本轮的 CPU 调度同时进行"""
    while True:
        # CPU: 处理上一轮结果 + 调度本轮
        # GPU: 同时在跑本轮的 forward
        # → CPU 和 GPU 的空闲时间互相填充
```

**图示对比**:

```
Normal (串行):
CPU: [调度][等待GPU][处理结果][调度][等待GPU][处理结果]
GPU:       [forward]                [forward]
                    ↑ GPU 空闲         ↑ GPU 空闲

Overlap (重叠):
CPU: [调度][处理上轮结果+调度本轮][处理上轮结果+调度本轮]
GPU:       [forward batch N      ][forward batch N+1    ]
           ↑ CPU 和 GPU 同时工作!
```

| | event_loop_normal | event_loop_overlap |
|---|---|---|
| **何时使用** | `--disable-overlap-schedule` | 默认模式 |
| **优点** | 简单，易调试 | 吞吐更高 (CPU/GPU 并行) |
| **缺点** | GPU 有空闲时间 | 逻辑复杂，需要 CUDA stream 管理 |
| **学习建议** | **先看这个** | 理解 normal 后再看 |

**关键实现细节**: overlap 模式需要用 CUDA Stream 隔离调度和 forward，并用 WAR (Write-After-Read) barrier 避免数据竞争。

---

## Q5: ForwardMode 各值含义

```python
class ForwardMode(IntEnum):
    EXTEND = ...       # Prefill: 第一次处理完整 prompt
    DECODE = ...       # Decode: 已 prefill，逐 token 生成
    IDLE = ...         # 空闲: 没有请求要处理
    
    # 投机解码相关:
    DRAFT_EXTEND = ...      # Draft 模型的 prefill
    TARGET_VERIFY = ...     # Target 模型验证 draft 的猜测
```

**触发条件**:

| Mode | 触发条件 | 出现频率 |
|---|---|---|
| EXTEND | waiting_queue 有新请求 | 每个请求 1 次 |
| DECODE | running_batch 有请求且无新请求 | 每个请求 N 次 (N=生成长度) |
| IDLE | 无请求 | 空闲时 |
| DRAFT_EXTEND | 投机解码, draft 模型处理新请求 | 有投机解码时 |
| TARGET_VERIFY | 投机解码, target 模型验证 | 每次 verify 1 次 |

---

## Q6: Chunked Prefill vs 普通 Prefill

```
普通 Prefill:
  请求 prompt = 2048 tokens
  → 一次性 forward 2048 tokens
  → 耗时: ~190ms (A100)
  → 问题: 这 190ms 内所有 decode 请求都在等待!

Chunked Prefill:
  请求 prompt = 2048 tokens, chunk_size = 512
  → 分 4 次 forward: 512, 512, 512, 512
  → 每次耗时 ~48ms
  → 每次 forward 还可以混入 decode 请求 (Mixed Batch)!
  → Decode 请求的延迟从 190ms 降到 48ms
```

**本质**: 把一个大 prefill 切碎，与 decode 交替执行，降低 decode 的"被抢占"延迟。

---

## Q7: TP vs PP vs DP — 三种并行的区别

```
TP (Tensor Parallelism): 切模型的"宽度"
  一个 Attention 层的权重 W 切成 4 份，分给 4 张 GPU
  每张 GPU 算一部分，然后 AllReduce 合并
  → 单请求延迟不变，但能跑更大模型
  → 通信开销: 每层 2 次 AllReduce

PP (Pipeline Parallelism): 切模型的"深度"
  32 层 Transformer 分成 4 段，每段 8 层放一张 GPU
  请求依次流过 4 张 GPU
  → 单请求延迟 ×4 (串行)，但可以流水线
  → 通信开销: 段间传 activation (一次)

DP (Data Parallelism): 复制模型，分请求
  4 张 GPU 各放一份完整模型
  请求被均匀分配到 4 张 GPU
  → 单请求延迟不变，总吞吐 ×4
  → 通信开销: 无!
```

| | TP | PP | DP |
|---|---|---|---|
| **切什么** | 每层切开 | 层间切开 | 不切，复制 |
| **适合** | 大模型单卡装不下 | 超大模型 | 模型能装进单卡 |
| **SGLang 典型** | TP=8 (单机 8 卡) | PP=2 (跨机) | DP=4 (4 个独立 Scheduler) |

---

## Q8: lock_ref 到底是什么？

`lock_ref` 是 RadixCache 树节点上的引用计数，防止正在使用的缓存被淘汰。

```
场景: 3 个请求共享前缀 [1, 2, 3]

请求 1 开始处理:
  match_prefix([1,2,3,4,5]) → 命中节点 [1,2,3]
  节点 [1,2,3] 的 lock_ref += 1  (现在 = 1)

请求 2 开始处理:
  match_prefix([1,2,3,6,7]) → 命中同一节点
  节点 [1,2,3] 的 lock_ref += 1  (现在 = 2)

内存不足，需要淘汰:
  节点 [1,2,3] lock_ref = 2 → 不能淘汰! (有活跃请求在用)
  节点 [4,5] lock_ref = 1 → 不能淘汰 (请求 1 在用)
  节点 [6,7] lock_ref = 1 → 不能淘汰 (请求 2 在用)
  → 只能等请求完成后释放

请求 1 完成:
  节点 [4,5] lock_ref -= 1  (现在 = 0) → 可以被淘汰了!
  节点 [1,2,3] lock_ref -= 1  (现在 = 1) → 仍被请求 2 引用，不能淘汰
```

**类比**: 就像图书馆的借阅系统。书被借出时不能从书架撤走，归还后才可以。

---

## Q9: CUDA Graph 是什么？为什么 SGLang 大量使用？

```
问题: GPU kernel 的"发射开销"

普通 forward:
CPU: [准备参数][发射kernel1][等待][发射kernel2][等待][发射kernel3]...
GPU:            [kernel1]           [kernel2]           [kernel3]

每次发射开销 ~5-10μs，一次 forward 有 ~100 个 kernel
总开销: 100 × 10μs = 1ms (decode 总共才 8ms!)

CUDA Graph:
  第一次: 录制所有 kernel 的执行序列
  之后: 一次性 replay 整个序列
CPU: [replay]
GPU: [kernel1][kernel2][kernel3]...  (无间隔!)

节省: ~1ms → decode 延迟从 8ms 降到 7ms (提升 12%)
```

**限制**: CUDA Graph 要求每次 forward 的 tensor shape 相同。所以 SGLang 需要对不同 batch size / seq_len 预先录制多份 graph (padding 到固定 shape)。

---

## Q10: Detokenizer 为什么是单独进程？

直觉上，detokenize 就是一次 `tokenizer.decode()` 调用，很快啊，为什么要单独进程？

**原因 1: CPU 开销不可忽略**

```
streaming 模式: 每个 token 都要 decode 一次
  50 个并发请求 × 每秒 100 tokens = 5000 次 decode/秒
  每次 decode 耗时 ~50μs
  总 CPU 开销 = 250ms/s (25% CPU!)
```

**原因 2: 不能阻塞 Scheduler**

如果 detokenize 在 Scheduler 进程中做:
```
Scheduler 循环:
  forward() → 8ms
  detokenize 50 个请求 → 2.5ms  ← 这段时间 GPU 空闲!
  下一轮 forward() → 8ms

把 detokenize 移出去:
  forward() → 8ms → 立即开始下一轮!
  GPU 利用率从 76% 提升到 ~100%
```

**原因 3: 增量 detokenize 的复杂性**

streaming 需要处理 UTF-8 边界问题:
```
token: [229, 184, 173]  → 这 3 个 token 拼起来才是 "中" 字
如果只 decode 前两个 → 乱码!
```

Detokenizer 进程维护了增量解码状态，处理这些边界情况。

---

## Q11: 为什么 Mac 上能 import SGLang 但不能真正推理？

在 Phase 1 中，我们用 `conftest.py` 给 `triton` 和 `sgl_kernel` 创建了 stub 模块。这意味着：

```python
import sglang                    # ✅ 成功 (Python 包结构完整)
from sglang.srt.managers.scheduler import Scheduler  # ✅ 成功 (Python 代码)

# 但是:
# 实际启动 Server 会失败，因为:
# 1. 没有 CUDA GPU → torch.cuda.is_available() = False
# 2. triton/sgl_kernel 是 stub → 真正的 CUDA kernel 无法执行
# 3. ModelRunner 加载模型权重需要 GPU 显存
```

**这是设计意图**：Phase 1 的目标是读懂架构和源码，不需要实际推理。所有 Mac 上能跑的单测 (如 `test_protocol.py`, `test_radix_cache_unit.py`) 都不涉及真正的 GPU 计算。

进入 Phase 2 后，在 GPU 机器上安装完整的 SGLang 就能真正推理了。参见 [gpu-setup.md](../setup/gpu-setup.md)。

---

## Q12: CUDA Graph 捕获失败的常见原因

CUDA Graph 把一组 GPU 操作"录制"下来，之后可以一次性"回放"，省去了逐个 kernel 的调度开销。但录制时有严格限制：

```
录制期间不能做的事:
❌ 动态分配显存 (torch.empty, torch.cat 等创建新 tensor)
❌ CPU-GPU 同步 (torch.cuda.synchronize, .item(), print(tensor))
❌ 改变 tensor 的 shape
❌ 条件分支依赖 GPU 数据 (if tensor.sum() > 0)
```

**SGLang 的处理方式**:
- Decode 阶段 batch 内每个请求只处理 1 个 token，shape 固定 → 适合 CUDA Graph
- Prefill 阶段 input 长度不固定 → 不用 CUDA Graph (或用 Chunked Prefill 固定 chunk 大小)
- `cuda_graph_runner.py` 预先录制多种 batch size 的 graph，运行时选最接近的

参见 [FAQ Q9](#q9-cuda-graph-是什么为什么-sglang-大量使用) 了解更多。

---

## Q13: GPU 显存不够怎么办？

显存主要被三部分占用：

```
GPU 显存 = 模型权重 + KV Cache + 激活值(Activations)

示例 (7B 模型, FP16):
  模型权重: 7B × 2 bytes = 14GB
  KV Cache: 取决于 batch size 和序列长度
  激活值:  取决于 chunked_prefill_size
```

**解决方案** (按优先级)：

| 方案 | 效果 | 怎么做 |
|------|------|--------|
| 换小模型 | 直接减少权重占用 | 用 1.5B/3B 替代 7B |
| 量化 | 权重占用减半或更多 | `--quantization fp8` 或 `awq` |
| 减小 `--mem-fraction-static` | 限制 KV Cache 池大小 | `--mem-fraction-static 0.7` |
| 减小 `--max-running-requests` | 限制并发数 | `--max-running-requests 16` |
| 减小 `--chunked-prefill-size` | 减少激活值峰值 | `--chunked-prefill-size 2048` |

**快速诊断**: 如果 OOM 发生在启动时（加载模型），说明模型太大；如果在运行时，说明 KV Cache 或并发数太高。
