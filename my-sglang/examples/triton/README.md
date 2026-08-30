# Triton kernel 入门：Docker CPU interpreter

本目录按“每次只增加一个核心概念”的顺序学习真实 `@triton.jit` kernel。默认使用 Docker
中的 Triton CPU interpreter 做数值验证：它不是 NumPy 模拟，不需要 NVIDIA GPU，但也不
生成 CUDA 代码，不能验证 GPU 性能。

```text
1D 索引                     二维行寻址                 二维 tile
vector add ──> fusion ──> softmax / RMSNorm ──> matmul ──> autotune ──> attention
   入门          入门             入门              入门       进阶          进阶
```

## 1. 学习目标

完成入门部分后，应能回答下面的问题：

| 问题 | 对应能力 |
|---|---|
| 一个 Triton kernel 怎么启动？ | 读懂 `@triton.jit`、`kernel[grid](...)` 和 meta-parameters。 |
| 一个 program 处理哪些数据？ | 根据 `program_id`、tile、logical lanes 和 offsets 算出全局下标。 |
| 尾块为什么不会越界？ | 根据 shape 构造 mask，并用于 `tl.load` / `tl.store`。 |
| 为什么要写自定义 kernel？ | 从 HBM 读写次数解释 fusion 的收益，而不是只看公式。 |
| 一行或一个矩阵块怎么处理？ | 使用 stride、reduction、二维 pointer 和 `tl.dot`。 |
| 结果和性能可信吗？ | CPU interpreter 只校验语义；不用它推断 GPU 性能。 |

CUDA 与 Triton 的完整概念映射见
[Triton 与 CUDA：真实 GPU kernel](../../docs/triton-cuda-basics.md)。
RMSNorm、Q/K/V、Attention 和 FFN 的公式与 shape 见
[Transformer 核心数学](../../docs/transformer-math.md)；这些课程在完整 block 中的位置见
[Triton 示例中的 Transformer 数据流](../../docs/triton-transformer.md)。

## 2. 前置条件

| 必需项 | 要求 |
|---|---|
| Docker | 能运行 `linux/amd64` 容器；Apple Silicon 会使用指令集模拟 |
| GPU | 不需要 |
| PyTorch | 容器内安装 CPU 版本 |
| Triton | 容器启动前设置 `TRITON_INTERPRET=1` |

### Docker：在 CPU 上运行全部课程

```bash
# 在仓库根目录 sglang/ 执行
docker build --platform linux/amd64 \
  -f my-sglang/docker/triton-interpreter.Dockerfile \
  -t my-sglang-triton:interpreter \
  my-sglang

docker run --rm --platform linux/amd64 my-sglang-triton:interpreter
```

容器入口 [`run-triton-lessons.sh`](../../docker/run-triton-lessons.sh) 会依次解释执行 01～07；
每课都与 PyTorch CPU reference 比较，任一 correctness 失败都会立即退出。成功标志是：

```text
PASS all Triton CPU-interpreter lessons
```

Interpreter 逐 op 执行 kernel，速度没有 GPU 意义。因此 Docker 流程不运行 benchmark；
02 和 06 中保留的 benchmark 代码只用于以后在真实 CUDA 环境学习性能测量。
06 在 interpreter 中固定使用第一组 config 验证计算；没有 GPU driver 就不声称完成了 autotune。

## 3. 学习路线

| 顺序 | 文件 | 状态 | 本课只增加什么 | 学完能回答 |
|---:|---|---|---|---|
| 01 | [`01_vector_add.py`](01_vector_add.py) | 已完成 | 1D grid、program、offset、mask、launch | 一个 Triton kernel 怎么启动？ |
| 02 | [`02_fused_elementwise.py`](02_fused_elementwise.py) | 已完成 | fusion、一次 HBM 读写、benchmark | 为什么需要自定义 kernel？ |
| 03 | [`03_row_softmax.py`](03_row_softmax.py) | 已完成 | stride、按行处理、`tl.max/sum`、片上数据 | 一个 program 怎么处理一整行？ |
| 04 | [`04_rmsnorm.py`](04_rmsnorm.py) | 已完成 | reduction + elementwise fusion、FP32 累加 | LLM 归一化 kernel 怎么写？ |
| 05 | [`05_matmul.py`](05_matmul.py) | 已完成 | 2D grid、二维 pointer、K tile、`tl.dot` | Triton 如何组织矩阵 tile？ |
| 06 | [`06_autotune_matmul.py`](06_autotune_matmul.py) | 已完成 | `@triton.autotune`、配置选择 | `BLOCK_SIZE`、`num_warps` 怎么选？ |
| 07 | [`07_attention.py`](07_attention.py) | 已完成 | online softmax、跨 tile 状态 | 单-query Attention 如何组合前面的能力？ |

01～05 是入门；06～07 是进阶。07 是强调数据流的教学实现，不是生产级 FlashAttention。

## 4. 为什么 vector add 后先学 fusion

`vector_add` 已经解决了一维 grid 和下标问题。下一课保持同样的索引，只把公式改为：

```text
out = silu(x + bias)
```

```text
多个独立算子：读 x,bias → 写 x+bias → 再读中间结果 → 写 silu 结果
Triton 融合：  读 x,bias → program 内完成 add 和 silu → 写最终结果
```

这一步第一次回答“为什么不直接调用现成 PyTorch 算子”：Triton 的价值不只是重写公式，
而是把多个操作放进一个 kernel，减少中间结果的 HBM 往返。同时引入正确的性能测量：

- `triton.testing.do_bench`
- GPU 异步执行与 warmup
- 有效带宽 GB/s
- correctness 与 performance 分开报告

学会 fusion 后再进入二维寻址和 reduction，可以避免一次同时理解太多概念。

## 5. 每个示例的统一结构

后续示例都按同一骨架组织：

```text
数学定义
   ↓
@triton.jit kernel
   ↓
Python launcher
   ↓
PyTorch reference
   ↓
correctness cases
   ↓
benchmark（需要时）
```

| 部分 | 职责 |
|---|---|
| 数学定义 | 先说清楚要计算什么，不从 API 开始。 |
| Kernel | 只包含设备侧数据读取、计算和写回。 |
| Python launcher | 检查 shape/device、分配输出、计算 grid、启动 kernel。 |
| PyTorch reference | 提供独立的正确结果，不能复用 Triton 的计算路径。 |
| Correctness | 覆盖正常 shape、不能整除 tile 的 shape、不同 dtype 和非法输入。 |
| Benchmark | warmup 后重复测量，并报告合适的 GB/s 或 TFLOPS。 |

## 6. 如何阅读一个 Triton 文件

不要从第一行逐字读到底。按下面顺序定位数据流：

```text
输入 shape / stride
        ↓
grid：启动多少个 program？
        ↓
program_id：当前是哪一个 program？
        ↓
tile / offsets：它负责哪些全局元素？
        ↓
mask：哪些元素有效？
        ↓
load → compute → store
        ↓
PyTorch reference 是否一致？
```

对每个 kernel，先手算一个具体 program。例如 `n=1003、BLOCK_SIZE=256、pid=3`：

```text
logical lanes = 0..255
offsets       = 3 * 256 + lanes = 768..1023
valid offsets = 768..1002
```

## 7. 运行示例

```bash
cd my-sglang
export TRITON_INTERPRET=1  # 必须在 Python 导入 Triton 前设置
python examples/triton/01_vector_add.py --n 19 --block-size 16 --num-warps 1
python examples/triton/02_fused_elementwise.py --n 19 --block-size 16 --num-warps 1
python examples/triton/03_row_softmax.py --rows 3 --cols 7 --num-warps 1
python examples/triton/04_rmsnorm.py --rows 3 --cols 7 --num-warps 1
python examples/triton/05_matmul.py --m 5 --n 7 --k 9 --block-m 16 --block-n 16 --block-k 16 --num-warps 1
python examples/triton/06_autotune_matmul.py --m 5 --n 7 --k 9
python examples/triton/07_attention.py --batch 1 --heads 2 --context 9 --head-dim 8 --block-n 8 --num-warps 1
```

01 的预期输出：

```text
PASS Triton vector-add kernel (runtime=cpu-interpreter, n=19, block_size=16, grid=2, num_warps=1)
```

这表示 Triton interpreter 输出通过了 PyTorch CPU reference 校验，不代表 GPU 可编译或性能最优。

## 8. 正确性验证

| 必测情况 | 目的 |
|---|---|
| shape 能整除 tile | 验证基本计算。 |
| shape 不能整除 tile | 验证 mask 和尾块。 |
| 最小合法 shape | 暴露 grid、offset 的边界错误。 |
| 支持的不同 dtype | 暴露精度和类型提升问题。 |
| 非法 shape / device | 让错误在 launcher 中明确失败。 |

正确性比较使用 PyTorch reference；不要因为两段代码长得相似就认为结果必然正确。

## 9. 性能测量规则

```text
首次调用：可能包含 JIT 编译时间，不计入稳定 kernel 时间
warmup：   让编译、缓存和 GPU 状态稳定
repeat：   重复测量，报告中位数及波动
metric：   elementwise/reduction 看 GB/s，matmul 看 TFLOPS
```

GPU launch 默认异步。不要用一次普通 Python 计时直接下性能结论；使用 CUDA event 或
`triton.testing.do_bench`，并确保测量边界包含正确的同步。

## 10. 常见误解

| 误解 | 正确理解 |
|---|---|
| 一个 Triton program 就是一个 CUDA thread | Program 描述一个数据 tile，编译器再映射到 warps/threads。 |
| 一个 `tl.arange` 元素就是一个 CUDA thread | 它是 logical lane，即 tile 内逻辑坐标。 |
| `BLOCK_SIZE=256` 就是启动 256 个 CUDA threads | 它首先表示一个 program 的逻辑 tile 大小。 |
| `grid=(4,)` 表示总共只有 4 个 CUDA threads | 它表示启动 4 个 Triton program instances。 |
| PyTorch reference 永远在 CPU 上运行 | 它跟随 tensor device；本 Docker 流程中是 CPU，CUDA tensor 环境中则是 GPU。 |
| correctness 通过就说明性能好 | 正确性和性能是两个独立结论。 |

## 11. 入门与进阶边界

| 入门阶段先掌握 | 进阶阶段再学习 |
|---|---|
| grid、program、tile、offset、mask | PTX / SASS 分析 |
| stride、reduction、fusion | occupancy、寄存器压力的硬件根因 |
| 二维 pointer、`tl.dot` | autotune 搜索空间设计 |
| 正确性与规范 benchmark | online softmax、paged attention、复杂持久化 kernel |

在理解 tile、访存和 reduction 前，不直接进入 Attention。Attention 会同时引入 Q/K/V
布局、矩阵乘、online softmax 和跨 tile 状态，无法判断问题究竟出在哪一层。

## 参考路线

Triton 官方也建议从简单教程按顺序学习：

1. [Vector Addition](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)
2. [Fused Softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)
3. [Matrix Multiplication](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)

本目录在 vector add 与 softmax 之间加入真实 fused elementwise，在 softmax 后加入
RMSNorm：前者降低学习跨度，后者把 reduction + fusion 连接到 LLM 常用算子。
