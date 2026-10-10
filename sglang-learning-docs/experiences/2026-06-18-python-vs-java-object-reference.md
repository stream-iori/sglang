# 对象引用：共享 Req，不等于复制请求

> 源码基准：learning `2aa70e3eb4`，已合入 origin/main `6fc8d9da32`（2026-10-10）。

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

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/managers/schedule_batch.py](../../python/sglang/srt/managers/schedule_batch.py) | filter_batch / merge_batch 与 Req 引用 |
