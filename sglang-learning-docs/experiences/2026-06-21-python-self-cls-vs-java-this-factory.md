# self 和 cls：实例与类

> 源码基准：上游 `16eb87c06f`；本地集成 `4f0e1cbea2`（2026-10-20）。本地日志增强与验证范围见总索引和 Mac 验证记录。

```python
class Item:
    def __init__(self, value):
        self.value = value
    @classmethod
    def make(cls, value):
        return cls(value)
```

| 名称 | 代表什么 |
|---|---|
| self | 这次调用绑定的实例，类似 Java this |
| cls | 这次 classmethod 绑定的类；可以是子类 |
| staticmethod | 不自动绑定实例或类 |

`ForwardBatch.init_new` 这样的名字表示工厂入口，但它具体是 classmethod 还是其他形式要看装饰器，不靠名字猜。工厂组织对象构造，与设备 Graph capture 是两码事。

## 子类工厂为什么用 cls

```python
class Base:
    def __init__(self, value):
        self.value = value
    @classmethod
    def make(cls, value):
        return cls(value)
class Child(Base):
    pass
assert isinstance(Child.make(3), Child)
```

写死 `Base(value)` 就不会自动构造 Child。与 Java static 工厂不同，classmethod 会绑定调用时的类；staticmethod 则不自动绑定 cls。

练习：看 ForwardBatch.init_new 的装饰器和 return，再解释它负责构造哪类数据。不要因为名字有 init 就把它当成 Python 语言的 __init__。

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/model_executor/forward_batch_info.py](../../python/sglang/srt/model_executor/forward_batch_info.py) | ForwardBatch.init_new |
