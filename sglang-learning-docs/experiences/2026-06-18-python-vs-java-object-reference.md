# 对象引用：共享 Req，不等于复制请求

> 源码基准：上游 `16eb87c06f`；本地集成 `4f0e1cbea2`（2026-10-20）。本地日志增强与验证范围见总索引和 Mac 验证记录。

```python
a = {'output_ids': []}
b = a
b['output_ids'].append(7)
assert a['output_ids'] == [7]
```

| 操作 | 含义 |
|---|---|
| b = a | 同一对象的另一条引用 |
| b = list(a_list) | 新列表，但内部元素仍可能是共享对象 |
| a is b | 是否同一对象 |
| a == b | 按类型定义比较是否相等 |

ScheduleBatch 的请求列表可以变化，Req 对象仍被多个结构引用。浅拷贝 batch 不代表其 Req/KV 生命周期已经隔离；修改输出列表时必须知道谁还持有引用。

## 浅拷贝的最小实验

```python
req = {'output_ids': []}
a = [req]
b = list(a)
assert a is not b
assert a[0] is b[0]
b[0]['output_ids'].append(42)
assert a[0]['output_ids'] == [42]
b.clear()
assert len(a) == 1
```

复制容器和复制元素是两件事，与 Java 的 List 浅拷贝有相似之处。SRT 的请求还关联设备资源，不能用 deepcopy 随意复制一个 Req 并期待 allocator/锁自动隔离。

验收：解释过滤 batch 后，旧记录为何仍可能持有同一请求；分别画 Python 引用与 KV 槽所有权，避免把两张图混在一起。

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/managers/schedule_batch.py](../../python/sglang/srt/managers/schedule_batch.py) | filter_batch / merge_batch 与 Req 引用 |
