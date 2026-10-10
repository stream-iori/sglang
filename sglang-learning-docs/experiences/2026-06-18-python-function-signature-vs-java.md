# 函数签名：位置、关键字与注解

> 源码基准：learning `2aa70e3eb4`，已合入 origin/main `6fc8d9da32`（2026-10-10）。

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

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/managers/tp_worker.py](../../python/sglang/srt/managers/tp_worker.py) | forward_batch_generation 的关键字参数 |
