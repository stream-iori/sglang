# 01、02、06 的分支：长输入、缓存与回收

配套大图：[F6 缓存所有权与回收](15-field-atlas.md)。

[返回全局地图](README.md) · 上一页：[模型](03-model.md) · 下一页：[overlap](05-overlap.md)

**这些分支都服务于同一条生成循环：减少重算、限制单轮工作量、让有限内存继续推进。**

```text
01 准入 --长 prompt--> 分块 EXTEND --补完--> 05 首输出
02 准备 --命中前缀--> 借用已存 KV --只算 suffix--> 04 模型
02 准备 --内存不足--> evict -> retract -> abort
06 结束 ------------> 解锁前缀 / 交接完整页 / 回收私有资源
```

## 分块：分的是已有 prompt，不是提前生成答案

沿用 prompt `[7,8]`，为了看见分块，把 chunk size 设为 1。假设首 token 不触发停止：

| 轮次 | 输入 | 本轮后 KV 对应 token | 状态 | output_ids |
|---:|---|---|---|---|
| 1 | EXTEND `[7]` | `[7]` | PREFILLING | `[]` |
| 2 | EXTEND `[8]` | `[7,8]` | RUNNING | `[10]` |
| 3 | DECODE `[10]` | `[7,8,10]` | RUNNING | `[10,11]` |

第一轮只读到 prompt 的一部分，此时预测的是该部分后面的 token，不能当成完整 prompt 的回答。当前教学实现只有一个 `chunked_req`，连续中间 prefill 的在途流水属于进阶标准 SRT 主题。

## 缓存：两条请求可以指向同一组 K/V

```text
已缓存 prefix [7,8] -------> [slot8 | slot9]  cache 持有
                                  ^
新请求 [7,8,9] 的 row ------+------+
             suffix 9 -----------> [slot12 | 空] 请求私有
```

新请求只算 `[9]`。旧 prefix 不计入 extend_len，但仍占内存。`lock_ref` 是实际保护引用计数；`cache_protected_len` 记录请求受保护的长度，不是单独执行锁操作的字段。

page_size=2 时，已提交长度 3 只有前 2 个位置构成完整页，尾部 1 个位置不能作为完整缓存页交接。完整页还必须属于可靠的已提交内容；“完整”不等于任何时候都可随便共享。

## 内存压力：先释放可丢的，再撤回能重建的

| 动作 | 释放什么 | 保留什么 | 下一步 |
|---|---|---|---|
| evict | 无锁 cache leaf page | 活跃请求资源 | 重试分配 |
| retract | victim 的 row、私有 KV、runner 状态，解除借用锁 | prompt、已确认 output_ids、已有缓存前缀可继续由 cache 持有 | 回 waiting 重新 EXTEND |
| abort | 无法推进请求的活跃资源 | 终态与原因 | 结束请求 |

```text
retract 前：prompt=[7,8]，output_ids=[10,11]
                     |
释放物理状态，保留逻辑序列
                     |
再次 admission：get_fill_ids()=[7,8,10,11]
                     |
重新 EXTEND 恢复上下文 -> 生成后续 token；不重复确认 10、11
```

`retracted_stain` 记住曾经撤回，下一次准入按全部剩余输出做更保守预留；它不增加 max_new_tokens。

## 06 正常结束：请求结束，不等于所有缓存必须消失

同步路径会将已提交完整页交给启用的 radix cache，释放私有尾页和 row，并移除 runner 请求状态。缓存里的无锁页可被后续请求复用，也可被淘汰。未启用 cache 时，不会保留这些页。

自检：row 被回收后，旧 slot 编号还能直接给重新准入的请求使用吗？不能假定仍有效；必须重新匹配缓存、分配并建立地址映射。

深入：[动态流程](../dynamic-flows.md)。源码：[`_cache_unfinished_req / _retract_req / _release_active_memory`](../../src/my_sglang/scheduler.py)。
