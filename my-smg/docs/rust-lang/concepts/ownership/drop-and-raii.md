# Drop 与 RAII：离开作用域如何收尾

[分类目录](index.md) · [总入口](../../index.md) · [问题索引](../../questions.md)

**拥有资源的值销毁时执行清理；作用域和所有权让成功、错误和取消走同一条释放路径。**

| 阅读准备 | 内容 |
|---|---|
| 前置 | [Move 与部分移动](move-and-partial-move.md) |
| 源码入口 | [src/worker.rs](../../../../src/worker.rs)：`impl Drop for InFlightGuard` |
| 阅读时机 | 按当前课程需要阅读，不据此跳过 learning 的阶段验收。 |

## 一个请求对应一个 guard

`src/worker.rs` 的节选：

```rust
pub struct InFlightGuard<'a> {
    worker: &'a Worker,
}

impl<'a> Drop for InFlightGuard<'a> {
    fn drop(&mut self) {
        self.worker.decrement_count();
    }
}
```

`begin_request` 先增加在途计数，再返回 guard。请求处理函数把它保存在 `_inflight_guard` 中；前导下划线只是表示有意不直接读这个变量，不会提前释放。

```text
begin_request → 计数 +1 → 持有 guard → 请求结束 → Drop → 计数 -1
                                      │
                         正常返回 / 错误返回 / 取消清理
```

| 写法 | 对清理的影响 |
|---|---|
| `let _inflight_guard = ...` | 保存 guard 到其作用域结束 |
| `drop(guard)` | 显式消费并销毁，立即执行清理 |
| `return` 或 `?` 提前返回 | 当前作用域中已初始化的局部资源仍会销毁 |
| Future 在暂停时仍被持有 | 存在 Future 状态里的局部资源仍活着 |
| Future 被丢弃 | 它拥有的局部状态被销毁 |

RAII 的意思是让资源的有效期跟随拥有值，而不是依靠每条业务分支手动清理。Drop 只能同步执行，不能在 drop 中直接 await；异步清理要另外显式等待。

本项目默认的进程终止和强制退出不保证执行普通作用域清理，不能把 RAII 当成进程退出时的可靠异步收尾。

## 自查与关联

为什么提前 return 也会让在途计数减一？

[Move 与部分移动](move-and-partial-move.md) · [共享借用与可变借用](borrowing.md) · [self、mut self、&self 与 &mut self](method-receivers.md)
