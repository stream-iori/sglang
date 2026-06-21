# Python Truthiness 对比 Java Boolean 判断

> **记录日期**: 2026-06-21
> **触发场景**: 阅读 Scheduler 中的 `if self.last_batch:`，不确定什么情况下条件为真，以及空 `ScheduleBatch` 是否会被当成 `False`。
> **一句话结论**: `if x` 会执行 Python 真值协议：先找 `__bool__()`，再找 `__len__()`，两者都没有时对象默认为真。当前 `ScheduleBatch` 没实现这两个方法，所以 `if self.last_batch` 实际只是在判断它是不是 `None`，不代表 batch 中有请求。

---

## 1. `if x` 不是类型推断

Python：

```python
if x:
    do_something()
```

可以先理解成：

```python
if bool(x):
    do_something()
```

它不是根据变量注解推断真假，而是运行时对对象执行固定的 truthiness（真值）协议。

## 2. 真值协议的判断顺序

对于对象 `x`，Python 按以下顺序判断：

```text
1. x.__bool__() 存在？
   ├─ 是：使用它返回的 bool
   └─ 否
       ↓
2. x.__len__() 存在？
   ├─ 是：长度为 0 → False，非 0 → True
   └─ 否
       ↓
3. 对象默认是 True
```

自定义 `__bool__()`：

```python
class Batch:
    def __init__(self, reqs):
        self.reqs = reqs

    def __bool__(self):
        return len(self.reqs) > 0


bool(Batch([]))       # False
bool(Batch(["req"]))  # True
```

只实现 `__len__()` 也可以参与真值判断：

```python
class Batch:
    def __init__(self, reqs):
        self.reqs = reqs

    def __len__(self):
        return len(self.reqs)
```

## 3. 常见 falsy 值

以下值会被判断为 `False`：

```python
None
False

0
0.0
0j

""
[]
()
{}
set()
range(0)
```

非零数字、非空容器和普通对象通常为 `True`：

```python
bool(1)          # True
bool(-1)         # True
bool("hello")    # True
bool([1])        # True
bool(object())   # True
```

“普通对象默认为真”是读业务代码时最容易忽略的一点。

## 4. SGLang 的 `self.last_batch`

Scheduler 初始化：

```python
self.last_batch: Optional[ScheduleBatch] = None
```

当前 `ScheduleBatch` 提供：

```python
def batch_size(self):
    return len(self.reqs)

def is_empty(self):
    return len(self.reqs) == 0
```

但它没有实现：

```python
__bool__()
__len__()
```

因此：

```python
if self.last_batch:
    ...
```

当前基本等价于：

```python
if self.last_batch is not None:
    ...
```

真值表：

| `last_batch` | `if self.last_batch` |
|---|---|
| `None` | `False` |
| 非空 `ScheduleBatch` 对象 | `True` |
| `ScheduleBatch(reqs=[])` | 仍然是 `True` |

示例：

```python
empty_batch = ScheduleBatch(reqs=[])

bool(empty_batch)       # True：普通对象默认真
empty_batch.is_empty()  # True：业务语义上确实为空
```

所以“对象存在”和“batch 非空”是两个不同问题。

## 5. 三种判断不要混用

### 判断 Batch 对象是否存在

```python
if self.last_batch is not None:
    ...
```

源码中的简写：

```python
if self.last_batch:
    ...
```

在当前 `ScheduleBatch` 实现下效果相同，但 `is not None` 语义更明确，也不受未来新增 `__bool__()` 的影响。

### 判断 Batch 是否包含请求

```python
if self.last_batch is not None and not self.last_batch.is_empty():
    ...
```

也可以判断内部列表：

```python
if self.last_batch is not None and self.last_batch.reqs:
    ...
```

### 判断变量是否恰好为布尔值 `True`

```python
if flag is True:
    ...
```

这比普通 `if flag:` 更严格，通常只在确实要区分 `True`、`False` 和其他 truthy/falsy 值时使用。

## 6. `and` / `or` 会短路

Scheduler 中的典型代码：

```python
if (
    not self.enable_hisparse
    and self.last_batch
    and self.last_batch.forward_mode.is_extend()
):
    ...
```

Python 从左向右判断：

```text
not enable_hisparse
        ↓ 为 True 才继续
self.last_batch
        ↓ 为 True 才继续
self.last_batch.forward_mode.is_extend()
```

如果 `self.last_batch is None`，第三个表达式不会执行，因此不会出现：

```text
AttributeError: 'NoneType' object has no attribute 'forward_mode'
```

这叫 short-circuit evaluation（短路求值）。

还要注意，Python 的 `and` / `or` 返回操作数本身，不一定返回 `bool`：

```python
None or "default"   # "default"
"value" and 123    # 123
[] or [1, 2]        # [1, 2]
```

只有放进 `if` 时，最终结果才继续接受真值测试。

## 7. 一个容易踩坑的特例：Tensor 和数组

不要假设所有容器都能直接放进 `if`。多元素 NumPy array 或 PyTorch Tensor 通常拒绝模糊判断：

```python
if tensor:
    ...
```

当 tensor 有多个元素时可能报错，因为系统不知道你想表达：

```text
所有元素为真？
任意元素为真？
Tensor 是否非空？
```

应该明确写出意图，例如：

```python
if tensor.numel() > 0:   # PyTorch：是否非空
    ...

if tensor.any():         # 是否至少一个元素为真
    ...

if tensor.all():         # 是否所有元素为真
    ...
```

## 8. 与 Java 对比

Java 的 `if` 条件必须是 `boolean`：

```java
if (lastBatch != null) {
    ...
}
```

Java 不能写：

```java
if (lastBatch) { }  // 编译错误
if (list) { }       // 编译错误
if (1) { }          // 编译错误
```

判断对象存在且集合非空：

```java
if (lastBatch != null && !lastBatch.isEmpty()) {
    ...
}
```

Python 对应：

```python
if self.last_batch is not None and not self.last_batch.is_empty():
    ...
```

或者借助列表 truthiness：

```python
if self.last_batch is not None and self.last_batch.reqs:
    ...
```

| 维度 | Python | Java |
|---|---|---|
| `if` 接受的条件 | 任意可执行真值测试的对象 | `boolean` / `Boolean` 拆箱结果 |
| 空列表 | `False` | 不能直接放入 `if`，必须 `isEmpty()` |
| 普通对象 | 默认 `True` | 不能直接放入 `if`，必须显式比较 |
| 空对象引用 | `None` 为 `False` | `null` 不能作为条件，写 `x != null` |
| 自定义真假 | `__bool__()` / `__len__()` | 返回 `boolean` 的显式方法 |

## 9. 最小记忆版

```text
if x 等价于对 x 做真值测试，不是类型推断。
顺序：__bool__ -> __len__ -> 普通对象默认 True。
None、零、空字符串、空容器是 False。
当前 ScheduleBatch 没有 __bool__/__len__，所以空 ScheduleBatch 仍然是 True。
if last_batch 判断对象存在；not last_batch.is_empty() 才判断 batch 中有请求。
Java 不支持 truthiness，if 条件必须显式得到 boolean。
```

延伸阅读：[Python `self` / `cls` 对比 Java `this` / 工厂模式](./2026-06-21-python-self-cls-vs-java-this-factory.md) · [Python 对象引用 vs Java](./2026-06-18-python-vs-java-object-reference.md)
