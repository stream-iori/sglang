# CUDA Graph、stream、padding 与请求边界

> 源码基准：learning `2aa70e3eb4`，已合入 origin/main `6fc8d9da32`（2026-10-10）。

CUDA Graph 把一段设备操作和依赖关系记录下来，后续更新输入数据后重放，减少 CPU 逐算子提交的开销。它仍需执行模型计算、读取本轮输入和 KV。

## Stream 和 Graph 不是 class 与 new

| 概念 | 类比 | 真正含义 |
|---|---|---|
| kernel | 一项工作 | GPU 上执行的函数，如矩阵乘、norm |
| stream | 一条工作队列 | 同一队列上的操作按规则排序，可和其他队列建立依赖 |
| Graph capture | 记下流程 | 捕获支持的操作、依赖和相关执行条件 |
| Graph replay | 再运行这份流程 | 提交图并执行记录下来的操作 |

Graph 捕获会使用 stream；Graph 不是 stream 的实例。SRT 还有 CPU 的 HTTP、分词、队列和调度，不能把整个服务塞进 GPU stream。

## “CPU 逐个提交 kernel”是什么意思

```text
普通执行
CPU：调用 norm ─ 调用 GEMM ─ 调用 RoPE ─ 调用 Attention ─ ...
GPU：    norm       GEMM        RoPE         Attention
        ↑ 每次调用有调度/launch 成本；提交可以异步

Graph
CPU：更新输入缓冲和元数据 ─ 提交一次 graph replay
GPU：                         norm → GEMM → RoPE → Attention → ...
```

CPU 在普通执行中提交的是本轮算子调用；“逐个提交已捕获 kernel”会混淆普通执行和重放。Graph 也不会免除全部 Python/CPU 工作，动态调度和不在图内的操作仍要执行。

## 为什么需要 padding

假设已捕获的 decode 桶大小为 8，本轮只有 5 个有效请求：

```text
固定缓冲：[ A B C D E pad pad pad ]  shape 仍为 8
有效数量：5
长度/索引/mask：指出哪些位置可以读写、哪些输出保留
```

捕获图通常要求可重放的 shape、地址和操作结构；用最近的兼容桶，可以避免为每个大小重新捕获。并非所有 Graph 都只有一个 batch 大小：可以捕获多个桶，也可以分段。

padding 还会用于 Attention backend、DP/TP collective 的对齐；即使 eager 执行，也可能需要它。

## Packed token 怎么知道边界

```text
真实请求长度：[3, 2]
累加边界：    [0, 3, 5]
打包 query：  [a a a b b]
               0     3   5
A = query[0:3]，B = query[3:5]
```

边界是输入元数据。固定 shape 的边界张量可以装入本轮的新值；Graph 固定张量的形状不等于固定其中所有数值。是否支持某种长度变化，仍取决于捕获路径、backend 和 eligibility 检查。

## 哪些部分难捕获

| 情况 | 困难来自哪里 |
|---|---|
| 读取 tensor 值后决定 Python 分支/循环次数 | 捕获时走的控制流未必适合下一轮 |
| 每轮新分配或更换地址 | 重放依赖的缓冲地址/生命周期变了 |
| 动态 shape、CPU 同步与回调 | 操作结构或捕获合法性改变 |
| backend 的 planning 或不支持 capture 的库调用 | 需要放在图外或换兼容实现 |
| 特殊多模态、稀疏、分布式或模型路径 | 需要专门适配；不能一概断言支持 |

“长度不同”不自动意味着难捕获；许多变化能用静态缓冲加动态元数据表达。真正要看的是操作能否在捕获条件下安全重放。

分段执行可以让兼容片段重放，把 planning/动态控制留在外面。分段 Graph、chunked prefill、按层 SPLIT_PREFILL 是不同概念。

## 当前源码怎么选

| 场景 | 当前可走路径 | 收益直觉 |
|---|---|---|
| 兼容的短 decode | decode Graph | 小算子多，launch 开销占比可能较大 |
| 兼容的 extend/prefill | prefill Graph，含不同捕获方案 | 依赖模式、backend、shape 桶和配置 |
| 不满足 can_run_graph | eager fallback | 省下 capture/padding 成本，保持动态能力 |
| 长 prefill、大 GEMM | 两者都可能 | 计算本身较长，省 launch 的相对收益可能小 |
| Mac MPS | 本地普通执行 | 没有 CUDA Graph |

简单估算：100 次 launch 每次假设 5µs，提交约 0.5ms。设备计算若 1ms，这部分占比明显；若 100ms，占比很小。数字只用于说明，不是本机测量。真正比较必须测总延迟、显存、capture 时间与 padding 后计算量。

## 当前 Graph 参数也按阶段拆分

| 参数 | 用途 |
|---|---|
| --cuda-graph-config | JSON 分别指定 decode/prefill 设置 |
| --cuda-graph-backend-decode | decode 的 full / breakable / disabled |
| --cuda-graph-backend-prefill | prefill 的 full / breakable / disabled |
| --cuda-graph-bs-decode / --cuda-graph-bs-prefill | 各阶段的捕获 batch 桶 |
| --cuda-graph-max-seq-len-prefill | prefill 允许的最大序列长度，超过则走 eager |

例如参数声明中的配置形式为：

```json
{"decode":{"backend":"full","max_bs":256},"prefill":{"backend":"breakable"}}
```

这是配置结构示例，不代表任意模型/backend 都支持该组合。源码声明的优先级是 JSON 配置高于阶段便捷参数和旧参数；有效值还受解析与设备规则影响。Mac 会禁用 CUDA Graph。

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/model_executor/model_runner.py](../../python/sglang/srt/model_executor/model_runner.py) | can_run_graph 与 fallback 条件 |
| [python/sglang/srt/arg_groups/fields/exec_.py](../../python/sglang/srt/arg_groups/fields/exec_.py) | Graph 两阶段参数声明与优先级 |
| [python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py](../../python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py) | decode 桶与 replay view |
| [python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py](../../python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py) | prefill capture 与 eligibility |
| [python/sglang/srt/model_executor/model_runner_components/cuda_graph_setup.py](../../python/sglang/srt/model_executor/model_runner_components/cuda_graph_setup.py) | 设备和配置初始化 |
| [python/sglang/srt/layers/attention/torch_native_backend.py](../../python/sglang/srt/layers/attention/torch_native_backend.py) | 有效长度与 query 切片 |
