# self、mut self、&self 与 &mut self

[分类目录](index.md) · [总入口](../../index.md) · [问题索引](../../questions.md)

**有没有 & 决定拥有还是借用，mut 决定绑定能否修改；这两件事分开看。**

| 阅读准备 | 内容 |
|---|---|
| 前置 | [共享借用与可变借用](borrowing.md)基础部分；项目应用另按节标题补前置。 |
| 源码入口 | [src/gateway/tests/support.rs](../../../../src/gateway/tests/support.rs)：`ManagedTask::cancel_and_join` |
| 阅读时机 | T02 读到基础自查；项目契约在阶段 3 后阅读。 |

## 先区分拥有、借用与修改权限

| 接收者 | 数据归谁 | 访问方式 |
|---|---|---|
| &self | 原拥有者 | 临时共享访问 |
| &mut self | 原拥有者 | 临时独占访问 |
| self | 按值交给方法 | 方法拥有收到的值；Copy 类型可以复制 |
| mut self | 与 self 一样按值接收 | 方法内部的 self 绑定可修改 |

Copy 是允许隐式复制的能力，整数通常可以复制，String 的按值交接则会移动。
调用 mut self 方法不要求原绑定为 mut：交出值后，方法内部建立新的可变绑定。
let mut 控制绑定可变性；&mut 表达一次独占借用。内部可变性在阶段 7 再学。

## 基础自查

| 类型 | 问题 |
|---|---|
| 预测 | 调用 mut self 方法前，原绑定必须声明为 mut 吗？ |
| 解释 | & 与 mut 在接收者中各解决什么问题？ |
| 边界 | 非 Copy 值交给消费 self 的方法后还能再次使用吗？ |

先写下判断与原因，再用自己已有的代码验证；遇到疑问回查本篇，不需要提前读后续应用。

## 阶段 3 后：项目中的接收者契约

先读[trait](../types/traits-and-bounds.md)；任务与许可在阶段 8～10 阅读。

## 从项目三个签名看起


| 项目签名节选 | 接收者 | 作用 |
|---|---|---|
| Worker::id(&self) -> &str | &self | 只借用节点，返回文本视图 |
| Policy::select(&mut self, ...) -> Option<usize> | &mut self | 借用策略并推进轮询状态 |
| ManagedTask::cancel_and_join(mut self) | mut self | 接管管理对象，并修改其 Option 字段 |
| CircuitPermit::success(mut self, ...) | mut self | 消费许可，禁止原凭证被重复上报 |

这些是签名摘录，省略参数和函数体。没有 & 的 self 接管值；mut 只说明这个局部绑定可修改。

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
