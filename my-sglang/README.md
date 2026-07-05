# my-sglang

`my-sglang` 是一个紧凑的单进程 runtime，用来复刻 SGLang 的核心请求生命周期：

```text
Req -> prefill batch -> running batch -> decode batch -> finish -> release KV
```

它有意跳过分布式 serving、采样变体、logprob、LoRA 和多模态路径。
radix cache 和 overlap scheduling 只保留教学版核心路径。

## 运行测试

From this directory:

```bash
../python/.venv/bin/python -m pytest
```

## 运行本地 MLX 生成

```bash
PYTHONPATH=src:../python ../python/.venv/bin/python -m my_sglang.cli \
  --model-path ~/.modelscope/models/Qwen3-0.6B \
  --prompt "Hello" \
  --max-new-tokens 4 \
  --trace
```

## Neovim / Pyright

从这个目录打开 Neovim，让本地 `pyrightconfig.json` 成为 LSP root：

```bash
cd my-sglang
nvim .
```

这个配置会把 Pyright 指向 `src`、`tests`、相邻的 SGLang 源码树
`../python`，以及现有的 uv 环境 `../python/.venv`。
