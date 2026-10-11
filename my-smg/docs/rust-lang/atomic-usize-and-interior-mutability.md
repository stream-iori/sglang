# Rust：`AtomicUsize` 与内部可变性

## 结论

`my-smg` 的每节点在途请求数是一个会被多个请求同时更新的整数。`AtomicUsize` 允许通过
`&self` 原子地修改这个整数，不需要取得整个 `Worker` 的 `&mut self`。

```text
多个请求 ──共享借用──> Worker
                         └─ AtomicUsize：每次加减都是原子操作
```

这里的 `AtomicUsize` 拼写是 **Atomic**，不是 `Aomtic`。

## 为什么 `fetch_add` 只需要 `&self`

普通字段和原子字段的区别：

| 字段 | 修改时通常需要 | 原因 |
|---|---|---|
| `usize` | `&mut self` | 普通写入要求独占访问 |
| `AtomicUsize` | `&self` | 类型自身保证并发读写安全 |

这叫**内部可变性**：外部只拿到共享引用 `&Worker`，但字段提供受控的修改方法。
它不是“所有 `&T` 都能随便修改”，也不是跳过 Rust 的安全规则；安全性由字段类型负责。

`AtomicUsize::fetch_add`、`fetch_sub` 和 `load` 的接收者都是 `&self`。因此包装它们的
`Worker::increment_count`、`decrement_count`、`counter` 也可以用 `&self`。
这与 `Worker::set_status(&mut self, ...)` 不同：`status` 是普通字段，没有内部可变性。

## 三个操作分别做什么

```rust
use std::sync::atomic::{AtomicUsize, Ordering};

let count = AtomicUsize::new(0);
assert_eq!(count.load(Ordering::Relaxed), 0);

let old = count.fetch_add(1, Ordering::Relaxed);
assert_eq!(old, 0); // 返回修改前的值；此时 count 已经是 1

let old = count.fetch_sub(1, Ordering::Relaxed);
assert_eq!(old, 1); // 此时 count 回到 0
```

`load` 已经返回 `usize`；如果函数返回类型是 `usize`，不要在前面加 `&`，否则会变成
`&usize`。不需要旧值时，可在 `fetch_add` / `fetch_sub` 调用后加分号，丢弃返回值。

原子“加一”是不可分割的一次操作。多个请求同时加一，不会因为各自先读到同一个旧值而丢失更新。

## `Relaxed` 到底放松了什么

```text
Relaxed：仍保证这一个计数的读写是原子的
         不保证它与其他内存读写的先后关系
```

在途计数只用于统计，不用它的值来判断其他字段是否已经更新，所以这里可以使用
`Ordering::Relaxed`。如果把计数用作“其他数据已经准备好”的信号，不能直接照搬这个选择；
那需要重新设计并发同步关系。

## 与 `Arc`、`Mutex` 的分工

| 类型 | 解决的问题 | 当前例子 |
|---|---|---|
| `Arc<T>` | 多个请求共同拥有状态 | 共享 `GatewayState` |
| `AtomicUsize` | 并发修改一个整数 | 每个 `Worker` 的在途计数 |
| `Mutex<T>` | 独占访问一组需要一起修改的数据 | 当前轮询策略的 `next` |

原子类型适合简单的独立数值操作。若必须同时检查、修改多个字段并保持整体一致，通常需要锁或
重新设计状态；把多个原子变量放在一起并不会自动产生一个“原子事务”。

`fetch_sub(1, ...)` 在计数为 `0` 时会下溢，不能无条件调用。当前练习先约定“加一后才减一”；
随后用 guard 的创建与销毁把两者配对，保证成功、失败和请求取消时都能归零。

## 继续阅读

- [Tokio 异步任务与取消测试](tokio-basics-and-task-cancellation.md)：等待计数到达 1，再取消任务，确认 guard 清理完成。
- [guard 借用 Worker：生命周期如何防止先销毁被借用者](ownership-move-and-borrowing.md#guard-借用-worker生命周期如何防止先销毁被借用者)：`InFlightGuard<'a>` 的借用约束。
- [`Mutex`、`MutexGuard`、解引用与毒锁](mutex-guard-deref-and-poisoning.md)：另一种安全的共享可变状态。
- [Trait、静态分发、动态分发与智能指针](traits-dispatch-and-smart-pointers.md)：原子计数如何让轮询策略的方法使用 `&self`。
