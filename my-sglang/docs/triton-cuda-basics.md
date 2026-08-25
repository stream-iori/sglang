# Triton 与 CUDA：真实 GPU kernel

这里的例子只讨论 **Linux + NVIDIA CUDA GPU** 上的 Triton。没有 CPU、NumPy 模拟或
interpreter 路径；没有 CUDA 设备就不能运行 kernel。

```text
Python launcher                 Triton compiler                  NVIDIA GPU
torch CUDA tensor ──launch──>   @triton.jit ──compile──>          grid
                                                                  └─ program instances
                                                                     └─ warps / threads
                                                                        └─ HBM load/store
```

## 先记住这张映射表

| CUDA 概念 | Triton 概念 | 不能误解成 |
|---|---|---|
| kernel launch | `kernel[grid](...)` | 普通 Python 函数调用 |
| grid | `grid` | 固定的 CUDA grid 维度；Triton 会继续决定底层 launch 细节 |
| block / CTA | 一个 program instance | 语言保证的一一映射 |
| `blockIdx.x` | `tl.program_id(0)` | 单个 CUDA thread id |
| thread | 无直接源代码等价物 | 一个 `tl.arange` 元素 |
| warp | `num_warps` 的主要控制对象 | 恒等于某个固定数量的 logical lanes |
| global memory | 指针与 `tl.load` / `tl.store` | Python/CPU 内存访问 |
| shared memory / registers | 编译器管理的 program 内数据与布局 | 可由 Python 变量名精确指定的位置 |

**核心边界：** `tl.arange(0, BLOCK_SIZE)` 创建的是一个逻辑 tile，不是在源码里创建
`BLOCK_SIZE` 个 CUDA threads。Triton 根据 GPU、数据类型、布局和 `num_warps` 将它降到
真实 warp/thread 指令。

## 真实例子：vector add

运行文件：[examples/triton/01_vector_add.py](../examples/triton/01_vector_add.py)。

```python
program_id = tl.program_id(axis=0)
offsets = program_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
mask = offsets < n_elements
x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
tl.store(output_ptr + offsets, x + y, mask=mask)
```

`n=1003, BLOCK_SIZE=256` 时：

```text
grid = (ceil_div(1003, 256),) = (4,)

program 0  -> offsets 0..255
program 1  -> offsets 256..511
program 2  -> offsets 512..767
program 3  -> offsets 768..1023  ── mask 只允许 768..1002 读写
```

| 源码 | CUDA 层面的意思 |
|---|---|
| `@triton.jit` | 声明由 Triton 编译、在 GPU 上执行的 kernel。 |
| `kernel[grid](...)` | 按 grid 启动 program instances。 |
| `tl.program_id(0)` | 当前 program 的一维工作编号。 |
| `tl.arange` | 形成该 program 要处理的逻辑元素向量。 |
| `mask` | 尾块屏蔽越界 lane；没有它会非法访问 HBM。 |
| `num_warps=4` | 给编译器的并行度提示，不改变数学结果。 |

## 运行条件与命令

条件：Linux、NVIDIA 驱动可见、CUDA 版 PyTorch，以及与其兼容的 Triton。请按 PyTorch
官方安装页选择与你的 CUDA/驱动匹配的安装命令；在该环境内再安装 Triton。

```bash
cd my-sglang
uv venv --python 3.12
source .venv/bin/activate
# 安装 CUDA 版 PyTorch（按你的 CUDA 版本选择官方命令）
uv pip install triton
python -c 'import torch; assert torch.cuda.is_available(), torch.cuda.is_available()'
python examples/triton/01_vector_add.py --n 1003 --block-size 256 --num-warps 4
```

预期：输出 `PASS real Triton CUDA kernel ...`。脚本将 Triton 输出与 PyTorch CUDA 的
`x + y` 对比；这验证数值正确性，不是性能结论。

## 性能要从硬件事实出发

```text
HBM 读取 x,y ──> program 内计算 ──> HBM 写回 out
                   ↑
          少一次中间 HBM 往返，fusion 才可能变快
```

| 可以用真实 GPU 实测 | 仍需 profiler/benchmark 才能下结论 |
|---|---|
| kernel 可编译、可启动、结果正确 | 最优 `BLOCK_SIZE` / `num_warps` |
| mask 是否正确处理尾块 | HBM 是否合并访问、是否带宽受限 |
| 不同配置的端到端耗时 | 寄存器压力、occupancy、warp 分歧的根因 |

测量时先 `torch.cuda.synchronize()`，做 warmup，再用 CUDA event 或 benchmark 工具多次测量。
不要把一次 Python 调用耗时、CPU interpreter 结果或 NumPy 对比结果当成 Triton 性能数据。
