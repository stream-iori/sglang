# Triton：真实 CUDA 示例

本目录只保留可在 NVIDIA CUDA GPU 上编译、启动的真实 Triton kernel。

| 文件 | 说明 |
|---|---|
| [`01_vector_add.py`](01_vector_add.py) | 一个 program 处理一个向量 tile：`grid`、`program_id`、logical lanes、mask、`num_warps`。 |

运行环境、Triton ↔ CUDA 映射和性能边界见
[Triton 与 CUDA：真实 GPU kernel](../../docs/triton-cuda-basics.md)。

```bash
cd my-sglang
python examples/triton/01_vector_add.py --n 1003 --block-size 256 --num-warps 4
```

没有可用 CUDA GPU 时，脚本会明确失败；不提供 NumPy、CPU interpreter 或 macOS 模拟替代。
