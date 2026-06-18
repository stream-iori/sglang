# Mac 环境调试指南 (已验证)

> macOS ARM64 上通过 MLX 后端 + 单元测试学习 SGLang。
> 本指南中的所有命令和测试均已在 Mac M 系列芯片上实际验证通过。
>
> **环境搭好后**: 跑一下 [动手实验 demo](../04-practice/exercises.md) 验证环境正常，然后开始 [Week 1](../01-architecture/foundations.md)。
> **调试进阶**: pdb 断点、VS Code 配置、火焰图等见 [debugging-guide.md](../05-reference/debugging-guide.md)。

---

## 一、一键搭建 (推荐)

```bash
cd /path/to/sglang
bash sglang-learning-docs/setup/setup_mac.sh
```

脚本会自动:
1. 创建 `.venv` (Python 3.10)
2. 安装 CPU-only PyTorch + SGLang 核心依赖
3. 验证所有核心模块可 import

---

## 二、手动搭建 (如需了解细节)

### Step 1: 创建 venv

```bash
cd /path/to/sglang
uv venv python/.venv --python 3.13
```

> 注意: venv 放在 `python/.venv` 是为了方便 IDE 索引 `python/` 目录下的源码。

### Step 2: 安装依赖

```bash
# CPU PyTorch
UV_HTTP_TIMEOUT=300 uv pip install --python python/.venv/bin/python \
  torch --index-url https://download.pytorch.org/whl/cpu

# SGLang 核心依赖
UV_HTTP_TIMEOUT=300 uv pip install --python python/.venv/bin/python \
  numpy pydantic fastapi pyzmq aiohttp requests pillow \
  "transformers==5.8.1" accelerate \
  pybase64 orjson msgspec interegular partial_json_parser "outlines==0.1.11" \
  IPython setproctitle packaging einops scipy tiktoken sentencepiece \
  prometheus-client psutil "openai>=1.0" torchvision \
  compressed-tensors gguf dill \
  mlx mlx-lm datasets uvicorn watchfiles uvloop soundfile python-multipart \
  pytest parameterized
```

### Step 3: 验证

```bash
PYTHONPATH="python" python/.venv/bin/python -c "
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey, TreeNode
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.managers.scheduler import Scheduler
print('All OK!')
"
```

---

## 三、可运行的单测 (已验证)

### Test 1: OpenAI Protocol — 17 tests ✅

```bash
PYTHONPATH="python" python/.venv/bin/python -m pytest \
  test/registered/unit/entrypoints/openai/test_protocol.py -v
```

**学到什么**: API 请求/响应的数据结构、字段校验、默认值

**对应学习周**: Week 1 (请求生命周期)

### Test 2: RadixCache — 单测 ✅

```bash
PYTHONPATH="python" python/.venv/bin/python -m pytest \
  test/registered/unit/mem_cache/test_radix_cache_unit.py -v
```

**学到什么**: RadixKey 操作、TreeNode 引用计数、前缀匹配、缓存淘汰、page 对齐

**对应学习周**: Week 2 (Scheduler 与 RadixCache)

### Test 3: 交互式探索数据结构

```bash
PYTHONPATH="python" python/.venv/bin/python
```

```python
# 1. 探索 SamplingParams
from sglang.srt.sampling.sampling_params import SamplingParams
params = SamplingParams(max_new_tokens=100, temperature=0.7, top_p=0.9)
print(vars(params))

# 2. 探索 RadixCache (真正的实现, 非 mock!)
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
cache = RadixCache.create_simulated(page_size=1)
cache.insert(RadixKey([1, 2, 3, 4]), None)
cache.insert(RadixKey([1, 2, 5, 6]), None)
cache.pretty_print()

result = cache.match_prefix(RadixKey([1, 2, 3, 4, 7, 8]))
print(f"hit_len = {len(result.device_indices)}")  # 4

# 3. 探索 ForwardMode
from sglang.srt.model_executor.forward_batch_info import ForwardMode
print(list(ForwardMode))  # EXTEND, DECODE, MIXED, IDLE, ...

# 4. 探索 OpenAI Protocol
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
req = ChatCompletionRequest(
    model="test",
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0.7,
)
print(req.model_dump_json(indent=2))
```

---

## 五、本地 MLX 服务与推理验证 (Apple Silicon)

在最新的官方主分支中，已原生支持 macOS (Apple Silicon M系列芯片) 的 MLX 后端推理。我们无需 GPU 即可在本地加载轻量模型进行端到端推理测试。

### 1. 下载 Qwen3-0.6B 轻量模型
使用 Modelscope 工具快速下载 Qwen3-0.6B 模型到 `~/.modelscope/models` 目录下：

```bash
# 激活环境并安装 modelscope（如果未安装）
source python/.venv/bin/activate
pip install modelscope

# 使用 python 脚本一键下载模型到 ~/.modelscope/models/Qwen3-0.6B 文件夹
python -c "
from modelscope import snapshot_download
model_dir = snapshot_download('Qwen/Qwen3-0.6B', local_dir='~/.modelscope/models/Qwen3-0.6B')
print('Model downloaded to:', model_dir)
"
```

### 2. 启动本地 MLX 推理服务
使用 `SGLANG_USE_MLX=1` 环境变量，配合 `--grammar-backend none` 禁用 CUDA/Triton 特有模块，在本地启动 SGLang MLX 服务器：

```bash
SGLANG_USE_MLX=1 PYTHONPATH="python" python/.venv/bin/python -m sglang.launch_server \
    --model ~/.modelscope/models/Qwen3-0.6B \
    --disable-cuda-graph \
    --grammar-backend none \
    --host 127.0.0.1 \
    --port 30000
```

服务启动成功后，会在控制台看到如下输出，且无任何 crash：
```text
[2026-06-17 23:09:43] INFO:     Uvicorn running on http://127.0.0.1:30000
[2026-06-17 23:09:44] MlxAttentionKVPool: 52838 slots x 28 layers x 8 heads x 128 dim
[2026-06-17 23:09:44] The server is fired up and ready to roll!
```

### 3. 本地客户端请求验证
在另一个终端窗口中，使用 `curl` 命令行工具向本地服务器发送生成请求：

```bash
curl http://127.0.0.1:30000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The capital of France is",
    "sampling_params": {
      "max_new_tokens": 10
    }
  }'
```

**预期响应**:
```json
{"text":" Paris. The capital of France is also the capital", ...}
```
这表明本地的 SGLang + MLX 服务已成功端到端运行！

---

## 六、远程 GPU Server 联调

本地 Mac 负责代码阅读 + 单测，远程 GPU 机器负责真正的推理服务。

### 远程机器: 启动 Server

```bash
# 安装 sglang (GPU 版)
pip install "sglang[all]"

# 启动 server
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --port 30000 \
    --host 0.0.0.0
```

### Mac 本地: 发请求

```bash
# 基础测试
curl http://<remote-ip>:30000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 10
    }'

# 查看服务器信息 (理解配置和状态)
curl http://<remote-ip>:30000/get_server_info | python -m json.tool

# 查看 metrics
curl http://<remote-ip>:30000/get_server_metrics
```

### Mac 本地: 用 Python SDK 测试

```python
from openai import OpenAI

client = OpenAI(base_url="http://<remote-ip>:30000/v1", api_key="none")

# 基础对话
resp = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "What is 2+2?"}],
    max_tokens=50,
    temperature=0.7,
)
print(resp.choices[0].message.content)

# 流式响应
stream = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Tell me a joke"}],
    max_tokens=100,
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

---

## 六、单测 → 源码学习的映射

| 单测文件 | 可运行 | 关联的核心源码 | 学习周 |
|---|---|---|---|
| `test_protocol.py` | ✅ 17 pass | `entrypoints/openai/protocol.py` | Week 1 |
| `test_radix_cache_unit.py` | ✅ 34 pass | `mem_cache/radix_cache.py` | Week 2 |
| 交互式 SamplingParams | ✅ | `sampling/sampling_params.py` | Week 3 |
| 交互式 ForwardMode | ✅ | `model_executor/forward_batch_info.py` | Week 3 |
| 远程 Server 联调 | ✅ (需 GPU) | 全链路 | Week 1-4 |

---

## 七、IDE 配置

### VS Code

```json
// .vscode/settings.json
{
    "python.analysis.extraPaths": ["./python", "./sglang-learning-docs"],
    "python.defaultInterpreterPath": "./python/.venv/bin/python",
    "python.envFile": "${workspaceFolder}/sglang-learning-docs/.env"
}
```

```bash
# sglang-learning-docs/.env
PYTHONPATH=python
```

### PyCharm

1. Settings → Project → Python Interpreter → 选择 `python/.venv/bin/python`
2. Settings → Project → Project Structure → 添加 `python/` 和 `sglang-learning-docs/` 为 Sources Root

---

## 八、常用命令速查

```bash
# 激活环境
source python/.venv/bin/activate

# 跑 protocol 单测
PYTHONPATH="python" python -m pytest test/registered/unit/entrypoints/openai/test_protocol.py -v

# 跑 RadixCache 单测
PYTHONPATH="python" python -m pytest test/registered/unit/mem_cache/test_radix_cache_unit.py -v

# 交互式探索
PYTHONPATH="python" python

# 搜索类定义
grep -rn "class Scheduler" python/sglang/srt/

# 搜索函数调用链
grep -rn "match_prefix" python/sglang/srt/ --include="*.py"

# 统计文件行数 (大文件 = 核心文件)
find python/sglang/srt -name "*.py" -exec wc -l {} \; | sort -rn | head -20

# 查看热点文件 (最近被频繁修改的)
git log --oneline --since="2 months ago" --name-only | grep "python/sglang/srt/" | sort | uniq -c | sort -rn | head -20
```
