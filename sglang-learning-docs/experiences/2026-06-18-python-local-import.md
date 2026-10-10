# 局部 import：等真的需要时才加载

> 源码基准：learning `2aa70e3eb4`，已合入 origin/main `6fc8d9da32`（2026-10-10）。

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

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/launch_server.py](../../python/sglang/launch_server.py) | Ray 等可选分支 |
| [python/sglang/srt/managers/scheduler.py](../../python/sglang/srt/managers/scheduler.py) | 平台相关导入 |
