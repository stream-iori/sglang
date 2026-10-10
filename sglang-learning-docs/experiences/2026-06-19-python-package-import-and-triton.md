# 平台 import 问题：先找导入链

> 源码基准：learning `2aa70e3eb4`，已合入 origin/main `6fc8d9da32`（2026-10-10）。

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

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/managers/scheduler.py](../../python/sglang/srt/managers/scheduler.py) | 平台 import |
| [python/sglang/kernels/README.md](../../python/sglang/kernels/README.md) | 新 kernel 路径 |
