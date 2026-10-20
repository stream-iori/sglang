# 局部 import：等真的需要时才加载

> 源码基准：上游 `16eb87c06f`；本地集成 `4f0e1cbea2`（2026-10-20）。本地日志增强与验证范围见总索引和 Mac 验证记录。

```python
def run_optional_backend():
    from optional_package import backend
    return backend.run()
```

| 放在哪里 | 导入发生时机 |
|---|---|
| 模块顶层 | 加载模块时 |
| 函数/分支内 | 执行到该位置时 |

局部 import 可以减少无关平台的重依赖、避免部分循环依赖。它不能让缺包功能正常执行；真正走到该分支仍要依赖存在。

当前 launch_server 的 Ray 分支局部导入并给出专门错误。MPS 缺包则要追 Scheduler 的导入链，不能简单假定所有可选组件都延迟加载了。

## 为什么它不等同于每次重新加载

局部 import 发生在执行到那一行时；成功加载后仍使用 Python 的模块缓存。它主要改变依赖触发时机，不保证模块会被反复执行或隔离。

| 情况 | 结果 |
|---|---|
| 平台分支未执行 | 可以不加载该分支的重依赖 |
| 分支实际执行但包缺失 | 仍会失败，需处理依赖或配置 |
| 两个模块循环顶层导入 | 可能看到对方尚未完成的初始化 |

练习：从 launch_server 的 Ray 分支解释为什么普通 HTTP 启动不需要先创建 Ray 对象；然后对比 Scheduler 的 MPS mixin 顶层导入，指出两者依赖触发时机不同。

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/launch_server.py](../../python/sglang/launch_server.py) | Ray 等可选分支 |
| [python/sglang/srt/managers/scheduler.py](../../python/sglang/srt/managers/scheduler.py) | 平台相关导入 |
