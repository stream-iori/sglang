# self、mut self、&self 与 &mut self

[分类目录](index.md) · [总入口](../../index.md) · [问题索引](../../questions.md)

**有没有 & 决定拥有还是借用，mut 决定绑定能否修改；这两件事分开看。**

| 阅读准备 | 内容 |
|---|---|
| 前置 | [struct、enum、impl 与 Self](../values/struct-enum-and-impl.md) |
| 源码入口 | [src/gateway/tests/support.rs](../../../../src/gateway/tests/support.rs)：`ManagedTask::cancel_and_join` |
| 阅读时机 | 按当前课程需要阅读，不据此跳过 learning 的阶段验收。 |

## 从项目三个签名看起


| 项目签名节选 | 接收者 | 作用 |
|---|---|---|
| Worker::id(&self) -> &str | &self | 只借用节点，返回文本视图 |
| Policy::select(&mut self, ...) -> Option<usize> | &mut self | 借用策略并推进轮询状态 |
| ManagedTask::cancel_and_join(mut self) | mut self | 接管管理对象，并修改其 Option 字段 |
| CircuitPermit::success(mut self, ...) | mut self | 消费许可，禁止原凭证被重复上报 |

这些是签名摘录，省略参数和函数体。没有 & 的 self 接管值；mut 只说明这个局部绑定可修改。

```text
&self      → 临时共享借用 → 原拥有者仍在
&mut self  → 临时独占借用 → 借用结束后原拥有者继续使用
self       → 值交出去     → 非 Copy 原变量失效
mut self   → 值交出去     → 新绑定可以修改
```

调用方不需要把 task 声明为 mut 才能调用 mut self：它没有修改自己的绑定，而是交出整个对象。完整时序见任务清理导读。
对于 Copy 类型，按值接收可以复制；不能把 self 一概理解为原变量必然失效。


## 为什么共同签名使用 `&mut self`

当前 Policy::select 的 workers 参数为 `&[Arc<Worker>]`。下面去掉 Arc 的接口是同步阶段的简化示意，只比较 self 的借用方式。

`RoundRobin` 选择后必须修改计数器，因此它需要：

```rust
fn select(&mut self, workers: &[Worker]) -> Option<usize>
```

`FirstHealthy` 本身没有状态，本来只需 `&self`。但是实现同一个 trait 时，方法签名必须一致，
所以它也接收 `&mut self`。

这说明接收者类型也是契约的一部分：

```text
&self      调用期间共享借用实现
&mut self  调用期间独占可变借用实现
self       调用会取得实现的所有权
```

如果未来要让多个任务并发调用一个需要 `&mut self` 的策略，仅仅加 `Arc` 不够；还需要重新设计
状态，或使用 `Mutex` 等同步机制。`Arc<T>` 只解决共享所有权，不自动提供可变访问。

所有权与借用基础见[所有权、转移与借用](move-and-partial-move.md)。

## 自查与关联

调用方不写 let mut，为什么仍能调用 mut self 方法？

[Move 与部分移动](move-and-partial-move.md) · [共享借用与可变借用](borrowing.md) · [Copy 与 Clone：复制值还是共享句柄](copy-and-clone.md)
