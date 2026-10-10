# self 和 cls：实例与类

> 源码基准：learning `2aa70e3eb4`，已合入 origin/main `6fc8d9da32`（2026-10-10）。

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

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/model_executor/forward_batch_info.py](../../python/sglang/srt/model_executor/forward_batch_info.py) | ForwardBatch.init_new |
