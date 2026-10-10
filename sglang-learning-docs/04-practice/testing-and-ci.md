# 测试与 CI：目录说明主题，注册说明运行环境

> 源码基准：learning `2aa70e3eb4`，已合入 origin/main `6fc8d9da32`（2026-10-10）。

当前 CI 从 `test/registered/` 发现测试，测试文件注册硬件、阶段和 runner。不要沿用旧版目录判断某个测试一定是 CPU/GPU 用例。

```text
test/registered/<主题> → register_*_ci 信息 → test/run_suite.py → CI 阶段/runner
test/manual/          → 专门的本地或非 CI 实验
```

| 当前规则 | 依据 |
|---|---|
| 支持 unittest 与 pytest | test/README |
| CI 默认 failfast，附加 -f | 文件 main 入口契约 |
| CI 估时和阶段/runner 注册值使用 literal | runner 做 AST 收集 |
| unit 目录按运行时模块组织 | 单模块测试归属 |
| kernels/ops 与 kernels/benchmark | kernel 测试和性能测量分开 |

## 本地已经适配的基础检查

```bash
PYTHONPATH=python python/.venv/bin/python -m pytest -q \
  test/registered/mps/unit/test_mps_runtime.py \
  test/registered/unit/entrypoints/openai/test_protocol.py
```

这两组测试不等于全服务验证；真实模型 HTTP 检查通过 verify_mac.py。

## 根据修改扩大检查

| 修改 | 需要的检查 |
|---|---|
| 文档/命令 | 路径、符号、shell 语法、可运行样例 |
| 配置 | runtime_context / 参数解析相关单测 |
| 统一缓存 | 对应组件、锁、分配/驱逐和必要的平台测试 |
| 输出协议 | OpenAI 单测 + 真实流式/非流式响应 |
| Graph/kernel/多卡 | 对应硬件、数值正确性、相关 CI suite |

统一缓存单测中有些仍依赖平台扩展，不保证 Mac 可以直接跑所有文件。先看 import 和 fixture，缺少硬件的用例应明确未执行，而不是当作通过。

## 对照源码

| 文件 | 读什么 |
|---|---|
| [test/README.md](../../test/README.md) | 完整规范 |
| [test/run_suite.py](../../test/run_suite.py) | 发现与执行 |
| [test/registered/mps/unit/test_mps_runtime.py](../../test/registered/mps/unit/test_mps_runtime.py) | MPS runtime |
| [test/registered/unit/entrypoints/openai/test_protocol.py](../../test/registered/unit/entrypoints/openai/test_protocol.py) | 协议单测 |
| [test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py](../../test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py) | 统一缓存用例 |
