# 两个 __init__：包导入与对象初始化

> 源码基准：上游 `16eb87c06f`；本地集成 `4f0e1cbea2`（2026-10-20）。本地日志增强与验证范围见总索引和 Mac 验证记录。

| 写法 | 什么时候执行 | 作用 |
|---|---|---|
| 包目录的 __init__.py | 第一次导入该包时 | 初始化包、暴露入口；可能带导入副作用 |
| 类里的 def __init__(self, ...) | 创建实例时 | 初始化实例字段 |

```text
import a.b → 先执行 a/__init__.py，再加载 b
Obj(args)  → 分配对象，再执行实例 __init__
```

类的 __init__ 不是模块入口。排查平台 import 错误时，包的初始化链可能先于你想调用的函数执行。

## 构造与导入分别可能失败

```python
class Request:
    def __init__(self, rid):
        self.rid = rid
r = Request('A')
assert r.rid == 'A'
```

这是实例初始化，与包的 `__init__.py` 无关。导入某个模块时，可能在定义类之前就因父包依赖失败，所以“我没有创建 CUDA 类”不代表不会触发平台 import。

自测：import 同一模块两次会无条件重跑全部顶层代码吗？答案：普通 import 使用 sys.modules 缓存；这不等于不同子进程共享模块状态。Scheduler/ModelRunner 的初始化还按组件 helper 装配，不能把类构造理解成一个简单字段赋值函数。

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/managers/scheduler.py](../../python/sglang/srt/managers/scheduler.py) | Scheduler 初始化方法 |
| [python/sglang/__init__.py](../../python/sglang/__init__.py) | 包初始化 |
