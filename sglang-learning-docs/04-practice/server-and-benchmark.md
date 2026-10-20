# 服务实操：先验证正确，再测性能

> 源码基准：上游 `16eb87c06f`；本地集成 `4f0e1cbea2`（2026-10-20）。本地日志增强与验证范围见总索引和 Mac 验证记录。

先通过可重复的功能检查，再谈吞吐或延迟。当前本地启动入口是 Torch MPS 脚本。

```bash
bash sglang-learning-docs/setup/launch_mac.sh
# 另一终端
python/.venv/bin/python sglang-learning-docs/setup/verify_mac.py \
  --url http://127.0.0.1:30000 --output /tmp/sglang-local-check
```

| 验证 | 为什么 |
|---|---|
| /health | 服务可接收请求 |
| /server_info | 实际设备与配置 |
| 两次 deterministic /generate | 输出稳定性和 prefix 命中 |
| OpenAI chat | template/协议与 engine 衔接 |
| SSE + [DONE] | 流式拼接与结束边界 |

## 手动请求

```bash
curl --fail http://127.0.0.1:30000/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"The capital of France is", "sampling_params":{"temperature":0,"max_new_tokens":8}}'
curl -N --fail http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"default","messages":[{"role":"user","content":"2+2=?"}],"max_tokens":32,"temperature":0,"stream":true,"chat_template_kwargs":{"enable_thinking":false}}'
```

功能验证能看到成功结果；性能测试还要控制 prompt 长度、输出长度、并发、请求速率和缓存冷热。

## 当前 benchmark 入口

旧 `sglang.bench_serving` 是兼容入口，源码已迁到 `sglang.benchmark.serving`。

```bash
PYTHONPATH=python python/.venv/bin/python -m sglang.benchmark.serving --help
```

根据当前 help 构造负载，不保留旧参数复制粘贴。没有实际执行的 benchmark，不在讲义中给出虚构吞吐。

| 对照维度 | 固定项 |
|---|---|
| prefix hit | 同一输入 IDs、长度与 sampling |
| Graph vs eager | 同硬件、模型、backend、负载；记录 padding/capture |
| 并发 | 同请求集合与输出上限 |
| backend | 功能/数值正确性先过，再测性能 |

本地记录见 [mac-validation](../setup/mac-validation.md)。历史上 30009/30010 成功验证，现已停止；需要服务时重新启动。

## 本轮验证脚本的完整范围

| 步骤 | 检查 |
|---|---|
| 来源 | Git HEAD、SRT 工作区 diff 指纹、Python/包版本、模型 config 指纹 |
| 有效配置 | mps、torch_native、pytorch、normal |
| 生成与复用 | 固定输出数、cold/warm ID 一致、重复请求有缓存命中 |
| 协议 | Chat 内容、流式拼接与非流式相同、DONE |
| 多请求 | A/B 的串行参考与并发结果分别一致 |
| 生命周期 | 两个请求结束后，新请求仍可执行 |
| 可选日志 | --status-dir 验证当次 run_id 的批次、请求行和 cache |

本脚本是功能验证，输出请求耗时不能替代基准测试。输出目录会写入输入输出内容，分享报告时仅提交需要的摘要，不把模型和虚拟环境加入 Git。

## 对照源码

| 文件 | 读什么 |
|---|---|
| [sglang-learning-docs/setup/verify_mac.py](../setup/verify_mac.py) | 真实验证内容 |
| [python/sglang/benchmark/serving.py](../../python/sglang/benchmark/serving.py) | 当前 benchmark 实现 |
| [python/sglang/bench_serving.py](../../python/sglang/bench_serving.py) | deprecated 兼容入口 |
