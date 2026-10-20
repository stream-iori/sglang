# if batch：对象存在与批次非空不同

> 源码基准：上游 `16eb87c06f`；本地集成 `4f0e1cbea2`（2026-10-20）。本地日志增强与验证范围见总索引和 Mac 验证记录。

| 写法 | 判断什么 |
|---|---|
| if x | __bool__，或 __len__，或默认真值 |
| x is None | 是否是 None |
| batch.is_empty() | batch 自己定义的请求集合是否为空 |

当前 ScheduleBatch 有 `is_empty()`，没有因此自动变成 bool 的空列表。类若未定义 __bool__/__len__，实例通常为真，即使里面的 reqs 是空的。

```text
None          → 没有批次对象
ScheduleBatch → 有对象，仍要按契约检查请求数/模式
IDLE batch    → 可能是合法的并行同步执行状态
```

所以不要把源码里的 `if batch` 一律改写为“请求数量大于零”；需要读返回 plan 的契约和对应循环。

## 真值的最小实验

```python
class Batch:
    def __init__(self):
        self.reqs = []
batch = Batch()
assert bool(batch) is True
assert len(batch.reqs) == 0
assert batch is not None
```

Java 的 if 通常要求 boolean；Python 会调用对象真值协议。`if batch` 与 `if batch.reqs` 不是同一个判断。

自测：空的 IDLE batch 能否有意义？答案：在某些并行同步路径中可以；要读调用契约，不能统一把空请求 batch 删除。更不能给类新增 __bool__ 来“修正”一个局部理解，因为会改变所有调用点。

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/managers/schedule_batch.py](../../python/sglang/srt/managers/schedule_batch.py) | is_empty |
| [python/sglang/srt/model_executor/forward_batch_info.py](../../python/sglang/srt/model_executor/forward_batch_info.py) | IDLE 模式 |
| [python/sglang/srt/managers/scheduler.py](../../python/sglang/srt/managers/scheduler.py) | normal loop 的 if batch |
