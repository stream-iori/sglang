# GPU 环境搭建指南

> **适用阶段**: Phase 2 开始前 (Week 5 之前完成)
> **预计用时**: 1-2 小时
> **前提**: 你已完成 Phase 1 (Week 1-4) 的 Mac 源码学习

---

## 一、硬件要求

| GPU | 显存 | 推荐学习模型 | 备注 |
|-----|------|-------------|------|
| T4 | 16GB | Qwen2.5-1.5B-Instruct | 入门够用 |
| A10 / L4 | 24GB | Qwen2.5-7B-Instruct | 推荐配置 |
| A100 / H100 | 40-80GB | Llama-3-8B / 70B (TP) | 可做分布式实验 |

**最低要求**: 1 张 NVIDIA GPU，16GB 显存，CUDA 12.1+

---

## 二、环境检查

### 2.1 确认 NVIDIA 驱动和 CUDA

```bash
# 确认驱动已安装
nvidia-smi

# 你应该看到类似输出：
# CUDA Version: 12.4
# GPU 0: NVIDIA A100-SXM4-80GB

# 确认 CUDA Toolkit
nvcc --version
# 如果没有 nvcc，可能需要安装 CUDA Toolkit
# 参考: https://developer.nvidia.com/cuda-downloads
```

### 2.2 确认 Python 版本

```bash
python3 --version
# 推荐 Python 3.10 - 3.12
```

---

## 三、安装 SGLang (GPU 模式)

### 3.1 创建虚拟环境

```bash
# 推荐用 uv (更快) 或 venv
python3 -m venv ~/.venvs/sglang-gpu
source ~/.venvs/sglang-gpu/bin/activate
```

### 3.2 从源码安装 (开发模式)

```bash
# 假设你已经 clone 了 sglang 仓库
cd /path/to/sglang

# 安装核心依赖 (GPU 模式)
pip install -e "python[all]"

# 安装 sgl-kernel (预编译 CUDA 算子)
pip install sgl-kernel
```

### 3.3 验证安装

```bash
# 验证 PyTorch 能看到 GPU
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"

# 验证 SGLang 能导入
python3 -c "import sglang; print('SGLang imported successfully')"
```

---

## 四、第一次启动 Server

### 4.1 选择模型

根据你的显存选择：

```bash
# 16GB GPU (T4) — 用 1.5B 小模型
MODEL="Qwen/Qwen2.5-1.5B-Instruct"

# 24GB GPU (A10/L4) — 用 7B 模型
MODEL="Qwen/Qwen2.5-7B-Instruct"

# 80GB GPU (A100) — 用 8B 或更大
MODEL="meta-llama/Llama-3.1-8B-Instruct"
```

### 4.2 启动 Server

```bash
# 最简启动命令
python3 -m sglang.launch_server \
    --model-path $MODEL \
    --port 30000

# 带更多控制参数的启动 (推荐学习时使用)
python3 -m sglang.launch_server \
    --model-path $MODEL \
    --port 30000 \
    --mem-fraction-static 0.8 \
    --log-level info
```

### 4.3 观察启动日志

启动后你会看到类似输出，**对照 Week 1 学的架构理解每一步**：

```
# 1. 加载模型权重 (对应 ModelRunner)
Loading model weights...
Model loaded in 12.3 seconds

# 2. 预热 CUDA Graph (对应 Week 3 FAQ Q9)
Warmup CUDA Graph...

# 3. Server 就绪 (对应 HTTP Server 进程)
The server is fired up and ready to roll!
```

### 4.4 发送第一个请求

```bash
# 用 curl 测试 (新开一个终端)
curl -s http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 64
  }' | python3 -m json.tool

# 或用 Python
python3 -c "
import openai
client = openai.Client(base_url='http://localhost:30000/v1', api_key='none')
resp = client.chat.completions.create(
    model='default',
    messages=[{'role': 'user', 'content': 'Hello!'}],
    max_tokens=64
)
print(resp.choices[0].message.content)
"
```

---

## 五、关键启动参数速查

| 参数 | 含义 | 示例 | 对应学习内容 |
|------|------|------|-------------|
| `--model-path` | 模型路径 (HuggingFace ID 或本地路径) | `Qwen/Qwen2.5-7B-Instruct` | Week 1: 模型加载 |
| `--tp-size` | Tensor Parallelism 并行度 | `--tp-size 2` (需 2 张 GPU) | Week 4: 分布式 |
| `--mem-fraction-static` | GPU 显存中分配给模型+KV Cache 的比例 | `0.8` (默认自动) | Week 2: 内存池 |
| `--max-running-requests` | 同时推理的最大请求数 | `64` | Week 2: 调度器 |
| `--chunked-prefill-size` | Chunked Prefill 分块大小 | `8192` | Week 3: Prefill |
| `--disable-radix-cache` | 关闭 RadixCache | (flag) | Week 2: RadixCache |
| `--log-level` | 日志级别 | `info` / `debug` | 调试 |
| `--port` | HTTP 服务端口 | `30000` | Week 1: HTTP Server |

---

## 六、Mac 远程连接 GPU Server

如果 GPU 在远程机器上，你可以从 Mac 通过 SSH 隧道访问：

```bash
# 在 Mac 上建立 SSH 隧道
ssh -L 30000:localhost:30000 user@gpu-server

# 然后在 Mac 上就可以用 localhost:30000 访问
curl http://localhost:30000/v1/models
```

或者使用 VS Code Remote SSH：
1. 安装 Remote-SSH 扩展
2. 连接到 GPU Server
3. 在远程终端启动 SGLang Server
4. 在本地浏览器访问 `localhost:30000`

---

## 七、常见问题

### Q: CUDA 版本不匹配

```
RuntimeError: The detected CUDA version (12.1) mismatches the version that was used to compile PyTorch (12.4)
```

**解决**: 安装匹配的 PyTorch：
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Q: OOM (显存不足)

```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**解决** (按优先级)：
1. 换更小的模型
2. 降低 `--mem-fraction-static`（如 `0.7`）
3. 减小 `--max-running-requests`
4. 使用量化：`--quantization awq` 或 `--quantization fp8`

### Q: 端口被占用

```
OSError: [Errno 98] Address already in use
```

**解决**:
```bash
# 找到占用端口的进程
lsof -i :30000
# 或换一个端口
python3 -m sglang.launch_server --port 30001 ...
```

### Q: 模型下载太慢

```bash
# 使用镜像 (如果在国内)
export HF_ENDPOINT=https://hf-mirror.com

# 或提前下载到本地
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --local-dir ./models/qwen-1.5b
python3 -m sglang.launch_server --model-path ./models/qwen-1.5b ...
```

---

## 八、环境验证清单

完成以下所有项目后，你就可以进入 Week 5 了：

- [ ] `nvidia-smi` 能看到 GPU
- [ ] `torch.cuda.is_available()` 返回 `True`
- [ ] `import sglang` 不报错
- [ ] SGLang Server 能成功启动
- [ ] 能收到 Server 的正确回复
- [ ] 理解启动日志中每一步对应的组件

---

> **下一步**: 进入 [Week 5: Server 启动与性能基准](./10-week5-server-and-benchmark.md)
