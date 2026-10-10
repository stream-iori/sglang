# 两个 __init__：包导入与对象初始化

> 源码基准：learning `2aa70e3eb4`，已合入 origin/main `6fc8d9da32`（2026-10-10）。

| 写法 | 什么时候执行 | 作用 |
|---|---|---|
| 包目录的 __init__.py | 第一次导入该包时 | 初始化包、暴露入口；可能带导入副作用 |
| 类里的 def __init__(self, ...) | 创建实例时 | 初始化实例字段 |

```text
import a.b → 先执行 a/__init__.py，再加载 b
Obj(args)  → 分配对象，再执行实例 __init__
```

类的 __init__ 不是模块入口。排查平台 import 错误时，包的初始化链可能先于你想调用的函数执行。

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/managers/scheduler.py](../../python/sglang/srt/managers/scheduler.py) | Scheduler 初始化方法 |
| [python/sglang/__init__.py](../../python/sglang/__init__.py) | 包初始化 |
