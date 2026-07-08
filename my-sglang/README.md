# my-sglang

`my-sglang` 是一个用于学习的单进程 mini runtime，用来复刻 SGLang 请求生命周期里的核心概念。
它不追求完整 serving 能力，而是把 scheduler、KV slot、radix cache、chunked prefill 和 MLX runner adapter 拆成容易阅读的小模块。

```text
Req
  -> prefill / chunked prefill
  -> running decode
  -> finish
  -> release or cache KV
```

## 当前能力

- 普通 prefill：waiting 请求会被组成 prefill batch，一次性写完 prompt KV。
- chunked prefill：通过 `chunked_prefill_size` 把长 prompt 拆成多个 prefill chunk，中间 chunk 只推进 KV，不产生用户可见 output。
- decode batch：已完成 prefill 的请求每轮 decode 一个 token。
- radix cache：缓存已完成请求的 prompt KV slot，新请求可复用命中的 prefix。
- overlap scheduling：教学版 MLX lazy start/kick/finalize 流程；当前不和 chunked prefill 组合。
- MLX adapter：可以复用相邻 SGLang 源码中的 `MlxModelRunner` 跑本地模型。
- trace：输出 JSON lines，便于观察 enqueue、prefill、decode、finish 等事件。

暂不覆盖：分布式 serving、复杂采样、logprob、LoRA、多模态、abort、真实分页 KV 张量管理、完整 SGLang admission policy。

## 学习路线

1. 读 `models.py`

   先理解 `Req`、`SamplingParams`、`RequestStatus` 和 `BatchForward`。重点看 `WAITING -> PREFILLING -> RUNNING -> FINISHED` 的状态含义，以及 `output_ids` 为什么只保存新生成 token。

2. 读 `pools.py`

   理解三个资源结构的关系：`ReqPool` 分配请求行号，`KVPool` 分配 token KV slot，`ReqToTokenMap` 建立 `(req_pool_idx, seq_pos) -> kv_slot` 映射。

3. 读 `scheduler.py`

   先看普通 prefill 和 decode，再看 chunked prefill。关键路径是：

   ```text
   add_request
     -> step
     -> _run_prefill / _run_chunked_prefill
     -> _run_decode
     -> _finish_req
   ```

4. 读 `radix_cache.py`

   关注压缩 radix 树如何保存 token prefix 到 KV slot 的映射。重点看 prefix 命中、插入分叉、节点 split。

5. 读 `runner.py`

   理解 scheduler 和模型执行之间的边界：`prefill` 用于第一块 prompt，`extend` 用于 chunked prefill 后续块，`decode_batch` 用于 running 请求。

6. 读 `overlap_scheduler.py`

   理解 MLX lazy execution 的 start/kick/finalize 拆分。这个模块用于学习 overlap scheduling，不是 chunked prefill 的第一入口。

7. 读 `tests/`

   `test_scheduler.py` 是最重要的学习材料，覆盖普通 prefill、chunked prefill、decode 混合、radix cache 复用和资源回滚。

## 运行测试

从 `my-sglang` 目录运行：

```bash
../python/.venv/bin/python -m pytest
```

## 运行本地 MLX 生成

普通 prefill：

```bash
PYTHONPATH=src:../python ../python/.venv/bin/python -m my_sglang.cli \
  --model-path ~/.modelscope/models/Qwen3-0.6B \
  --prompt "Hello" \
  --max-new-tokens 4 \
  --trace
```

chunked prefill：

```bash
PYTHONPATH=src:../python ../python/.venv/bin/python -m my_sglang.cli \
  --model-path ~/.modelscope/models/Qwen3-0.6B \
  --prompt "Explain chunked prefill in one sentence." \
  --max-new-tokens 4 \
  --chunked-prefill-size 8 \
  --trace
```

overlap scheduling：

```bash
PYTHONPATH=src:../python ../python/.venv/bin/python -m my_sglang.cli \
  --model-path ~/.modelscope/models/Qwen3-0.6B \
  --prompt "Hello" \
  --max-new-tokens 4 \
  --overlap \
  --trace
```

`--overlap` 和 `--chunked-prefill-size` 当前不能同时开启。

## Chunked Prefill 读法

开启 `chunked_prefill_size=N` 后，一个长 prompt 不会在一次 prefill 中全部写入 KV。请求会先进入 `PREFILLING`：

```text
step 1: prompt[0:N]      -> 写 KV，忽略临时 next token
step 2: prompt[N:2N]     -> 写 KV，忽略临时 next token
...
last:   prompt[k:end]   -> 写 KV，保留真正的 next token，进入 RUNNING 或 FINISHED
```

中间 chunk 之所以会产生临时 token，是因为底层模型每次 forward 都会给出 next-token logits。学习版会忽略这些临时 token，只在最后一个 prefill chunk 后把 token 加到 `Req.output_ids`。

## Neovim / Pyright

从这个目录打开 Neovim，让本地 `pyrightconfig.json` 成为 LSP root：

```bash
cd my-sglang
nvim .
```

这个配置会把 Pyright 指向 `src`、`tests`、相邻的 SGLang 源码树 `../python`，以及现有的 uv 环境 `../python/.venv`。
