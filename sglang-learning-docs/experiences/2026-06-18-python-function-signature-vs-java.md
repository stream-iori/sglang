# 函数签名：位置、关键字与注解

> 源码基准：上游 `16eb87c06f`；本地集成 `4f0e1cbea2`（2026-10-20）。本地日志增强与验证范围见总索引和 Mac 验证记录。

```python
def build(value: int, *, enabled: bool = True) -> str:
    return str(value) if enabled else ''
build(3, enabled=False)
```

| 语法 | 含义 |
|---|---|
| : int / -> str | 类型提示，通常不自动做运行时验证 |
| 默认值 | 调用时未传就使用它；避免可变默认对象 |
| * 后的参数 | 只能按名称传 |
| / 前的参数 | 只能按位置传 |
| Callable | 可以调用的对象/函数的类型描述 |
| Optional[T] | 允许 T 或 None，不表示“可以不传参数” |

读当前 ModelRunner/worker 的关键字参数，要区分弃用兼容参数和当前 metadata 契约，不能只看函数名相同就认为调用方式没变。

## Java 对照与调用练习

Java 常用重载或 builder 表达可选参数；Python 常用默认参数和 keyword-only 参数。注解不等于编译期或运行时强制验证，接口边界可能另用 Pydantic 等机制检查。

```python
def allocate(count, /, *, pool=None):
    return count, pool
assert allocate(2, pool='full') == (2, 'full')
# allocate(count=2) 或 allocate(2, 'full') 会违反签名。
```

自测：`Optional[int]` 的参数没有默认值，能省略吗？答案：不能；允许 None 与可以不传是两回事。读取 worker 新签名时逐项检查调用，不靠类型名猜。

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/managers/tp_worker.py](../../python/sglang/srt/managers/tp_worker.py) | forward_batch_generation 的关键字参数 |
