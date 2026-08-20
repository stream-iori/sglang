# Triton 基础入门：CPU 语义模拟

本目录用 NumPy 模拟 Triton 的**数据索引语义**。它不安装 Triton，不运行 GPU，
也不能测量 CUDA 性能。目标是先看清 kernel 如何把数据分给各个 program，再去读真实
Triton 代码。

```text
真实 GPU:     grid -> program -> logical lanes -> HBM load/store
本目录 CPU:   for pid -> offsets -> NumPy masked load/store
```

## 学习顺序

| 顺序 | 示例 | 新概念 |
|---:|---|---|
| 1 | `01_vector_add.py` | grid、`program_id`、offsets、mask |
| 2 | `02_vector_fusion.py` | 一个 program 内的算子融合 |
| 3 | `03_row_softmax.py` | 一行一 program、max/sum reduction |
| 4 | `04_rmsnorm.py` | reduction + fusion，LLM 常见小算子 |

先阅读 [CPU Triton 入门主文档](../../docs/triton-cpu-basics.md)，再运行：

```bash
cd my-sglang
uv run python examples/triton/01_vector_add.py --n 1003 --block-size 256 --trace
uv run python examples/triton/03_row_softmax.py --rows 3 --cols 257 --block-size 512 --trace
```

`--trace` 的一行代表一个逻辑 program。最后一个 program 的 `valid` 比 `offsets`
短，就是 Triton mask 防越界的原因。

```text
pid=3 offsets=768..1023 valid=768..1002
```

## 边界

| 本目录验证 | 本目录不模拟 |
|---|---|
| grid、program、offset、mask、归约、数值结果 | GPU 编译、thread/warp 映射、SRAM、同步、性能 |
| Triton 代码的逻辑正确性 | `BLOCK_SIZE` 或 `num_warps` 的真实最优值 |

每个脚本都会与等价 NumPy reference 比对，成功打印 `PASS`。测试命令：

```bash
uv run --extra test pytest tests/test_triton_examples.py -q
```

## 可选：真实 Triton CPU 后端

`triton-cpu` 是 Triton 官方组织维护的**实验性** CPU 后端；它可执行真实的
`@triton.jit` kernel，不是本目录的 NumPy 模拟。当前工作机是 macOS arm64，而上游
声明的支持平台是 Linux，因此不要在当前 `.venv` 中执行 `pip install triton` 并期待
CPU kernel 可用。

在独立的 Linux 环境中安装，避免污染 `my-sglang/.venv`：

```bash
git clone --recurse-submodules https://github.com/triton-lang/triton-cpu.git
cd triton-cpu
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r python/requirements.txt -r python/test-requirements.txt \
  -r python/tutorials/requirements.txt
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -e . --no-build-isolation
uv run python python/tutorials/01-vector-add.py
```

真实 CPU kernel 需要在运行前调用
`triton.runtime.driver.set_active_to_cpu()`，或设置 `TRITON_CPU_BACKEND=1`。
安装成功后再把本目录的 vector-add 扩展为一份可选的真实 Triton 对照实现。
