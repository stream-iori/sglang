# Apple Silicon Docker：交互式运行 Triton kernel

本教程在 Apple Silicon Mac 上通过 Docker 运行真实的 `@triton.jit` kernel。它使用
Triton interpreter：kernel 的 Triton 语义在 CPU 上执行，可用于学习和正确性验证；它
**不**生成 GPU 代码，也不能用于性能测试。

## 文件

- `docker/triton-learning.Dockerfile`：Python 3.12、CPU PyTorch、NumPy、Triton 和 `uv`。
- `examples/triton/05_interpreter_vector_add.py`：vector-add kernel，演示 `tl.program_id`、
  `tl.arange`、masked `tl.load` 与 `tl.store`。

镜像以已拉取的华为云基础镜像为基础：

```text
swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/python:3.12-slim-bookworm
```

基础镜像是 `linux/amd64`，而 Apple Silicon 是 `linux/arm64`，所以每个 Docker 命令
都必须指定 `--platform linux/amd64`。Docker Desktop 会进行模拟执行，首次构建较慢是正常的。
已验证的软件版本为 `torch==2.13.0+cpu`、`triton==3.7.1` 和 `numpy==2.5.2`。

## 构建镜像

在仓库根目录（`sglang/`）执行：

```bash
docker build --platform linux/amd64 \
  -f my-sglang/docker/triton-learning.Dockerfile \
  -t triton-learning:interpreter \
  my-sglang
```

## 进入容器并运行任意示例

仍在仓库根目录执行。此命令只挂载整个 `my-sglang` 工作区到容器的 `/workspace`，因此
源代码、文档和容器内创建的 `.venv` 都在同一个挂载目录中：

```bash
docker run --rm -it --platform linux/amd64 \
  -v "$PWD/my-sglang:/workspace" \
  -w /workspace \
  triton-learning:interpreter
```

容器内默认已有全局安装的 Triton、PyTorch 和 NumPy。可直接运行任意 Python 文件，例如：

```bash
python examples/triton/05_interpreter_vector_add.py
python examples/triton/05_interpreter_vector_add.py --n 1025 --block-size 256
```

预期输出：

```text
PASS real @triton.jit kernel via CPU interpreter (n=1003, block_size=256, grid=4)
```

退出容器：

```bash
exit
```

## 在挂载工作区中使用 uv 和 .venv

若希望项目级虚拟环境也保存在挂载的 `/workspace/.venv`，在**容器内**执行一次：

```bash
uv venv --system-site-packages .venv
source .venv/bin/activate
python examples/triton/05_interpreter_vector_add.py
```

`--system-site-packages` 使 `.venv` 复用镜像中已验证的 Triton/PyTorch 依赖，避免再次下载。

不要复用在 macOS 本机创建的 `.venv`：它通常包含 ARM64 二进制包，而这个容器运行
的是 AMD64 Linux。请只在上述容器中创建和更新 `my-sglang/.venv`。

## 边界与下一步

`TRITON_INTERPRET=1` 是调试/学习模式。要学习 `num_warps`、autotune、生成代码、访存和
真实性能，可把这些脚本移到 Linux + NVIDIA GPU 环境，并移除该环境变量后运行。
