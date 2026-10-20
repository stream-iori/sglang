# 平台 import 问题：先找导入链

> 源码基准：上游 `16eb87c06f`；本地集成 `4f0e1cbea2`（2026-10-20）。本地日志增强与验证范围见总索引和 Mac 验证记录。

```text
导入子模块
  → 父包 __init__
  → 顶层 imports
  → 平台判断分支 / 局部 imports
  → 首个缺失或不兼容依赖
```

| 证据 | 为什么需要 |
|---|---|
| 完整 traceback | 首个失败点和进入它的路径 |
| Python / package 版本 | 区分缺包与 API 不兼容 |
| 实际源码路径 | 避免查错安装包 |
| device / backend 配置 | 区分平台检测和实际执行选择 |

本次干净 MPS 环境确实因 Scheduler 导入的 mixin 需要 mlx.core 而失败。保留核心兼容包后，固定标准 Torch runner 并通过服务验证；不能把包存在等同于后端启用。

若是 Triton/CUDA 导入失败，同样按堆栈定位。当前 kernel 门面在 sglang.kernels，旧 jit_kernel 路径已经迁移。不要遇到一个 import 错误就安装所有 CUDA 依赖。

## 缺包问题的最小报告

记录 Python 路径、sglang 源码路径、首个 traceback、包版本、平台检测和实际 backend。把“平台可用性检测”与“模型选择执行后端”分开：mlx 核心包存在，只能证明导入可满足，不能证明模型由 mlx-lm 执行。

本教程启动强制 `SGLANG_USE_MLX=0`，用 server_info 和实际模型日志确认标准 Torch MPS。当前安装要求取自源码依赖，不复用旧环境的版本假设。

练习：为什么直接 pip 安装标准 CUDA 全依赖不能解决 Mac 的所有 import 问题？答案：平台、wheel 和硬件不匹配，且首个失败导入可能是另一条兼容链；应按证据修正确实需要的依赖。

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/managers/scheduler.py](../../python/sglang/srt/managers/scheduler.py) | 平台 import |
| [python/sglang/kernels/README.md](../../python/sglang/kernels/README.md) | 新 kernel 路径 |
