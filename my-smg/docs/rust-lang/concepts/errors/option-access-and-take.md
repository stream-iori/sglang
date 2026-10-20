# Option 中的借用与取值：as_ref、as_deref、take

[分类目录](index.md) · [总入口](../../index.md) · [问题索引](../../questions.md)

**借用整个容器、借用内部值和取走内部值是三种操作。**

| 阅读准备 | 内容 |
|---|---|
| 前置 | [Option 与 Result](option-and-result.md) · [借用](../ownership/borrowing.md) · [Box](../sharing/box.md)基础部分 |
| 源码入口 | [src/gateway/tests/support.rs](../../../../src/gateway/tests/support.rs)：handle.take() |
| 阅读时机 | 阶段 3 后按需阅读；任务应用在阶段 8～9。 |

## 用 Box 统一理解 `as_ref()`、`as_deref()` 与 `take()`

**`as_ref()` 借用盒子，`as_deref()` 借用盒子指向的整数，`take()` 把盒子取走。**
这里三个方法都调用在 `Option` 上。`Box<i32>` 拥有一个堆上的整数，解引用目标是 `i32`。

| 写法（`slot` 是 `Option<Box<i32>>`） | 得到的类型 | 做什么 | 原 `slot` |
|---|---|---|---|
| `&slot` | `&Option<Box<i32>>` | 借用整个 Option | 不变 |
| `slot.as_ref()` | `Option<&Box<i32>>` | 有盒子就借用盒子 | 不变 |
| `slot.as_deref()` | `Option<&i32>` | 有盒子就借用里面的整数 | 不变 |
| `slot.take()` | `Option<Box<i32>>` | 把盒子的所有权取出来 | 变成 `None` |

先比较两种借用：

```rust
let mut slot: Option<Box<i32>> = Some(Box::new(42));

let boxed: Option<&Box<i32>> = slot.as_ref();
let number: Option<&i32> = slot.as_deref();

println!("{boxed:?} / {number:?}");
println!("{slot:?}"); // 原值仍在，没有移动 Box，也没有复制整数
```

```text
slot：Some(Box ──拥有──> 42)
             ↑            ↑
as_ref() ────┘            │
得到 Some(&Box<i32>)      │
                          │
as_deref() ───────────────┘
得到 Some(&i32)
```

`Option` 这一层仍然保留：原来是 `Some`，返回包着引用的 `Some`；原来是 `None`，返回 `None`。
`as_ref()` 不等于 `&slot`：前者把引用放在 Option 里面，后者是引用整个 Option。

再接着取走盒子：

```rust
// 接上例；boxed、number 后面不再使用，前面的共享借用已结束
let extracted: Option<Box<i32>> = slot.take();

assert!(slot.is_none());
assert_eq!(extracted.as_deref(), Some(&42));
assert!(slot.take().is_none()); // 第二次已经没有盒子可取
```

```text
take 前：slot      = Some(Box ──拥有──> 42)

take 后：slot      = None
         extracted = Some(原来的 Box ──拥有──> 原来的 42)
```

这里移动的是原来的 Box，没有创建新盒子，没有复制堆上的整数。
`extracted.as_deref()` 只是借用整数做断言，仍然由 `extracted` 拥有 Box。
`slot` 本身没有被移动，所以还能使用，只是里面已经是 `None`。

| 方法 | 对接收者的要求 | 通用结果 |
|---|---|---|
| `as_ref()` | 共享借用 `&self` | `Option<T>` → `Option<&T>` |
| `as_deref()` | 共享借用 `&self`，且 `T` 实现 `Deref` | `Option<T>` → `Option<&T::Target>` |
| `take()` | 可变借用 `&mut self` | 返回原来的 `Option<T>`，给原位置留下 `None` |

因此 `slot` 需要声明为 `mut` 才能调用 `take()`；只调用前两者不需要。
如果后面还要使用 `boxed` 或 `number`，它们的共享借用仍然有效，就不能先调用 `take()`，
因为取走 Box 会让旧引用失去依托。

参见标准库 [Option::as_ref](https://doc.rust-lang.org/std/option/enum.Option.html#method.as_ref)、
[Option::as_deref](https://doc.rust-lang.org/std/option/enum.Option.html#method.as_deref) 与
[Option::take](https://doc.rust-lang.org/std/option/enum.Option.html#method.take)。

T::Target 表示解引用目标类型，本例是 i32；[关联类型](../types/associated-types.md)的通用语法可延后阅读。

## 基础自查

| 类型 | 问题 |
|---|---|
| 预测 | 连续执行两次 take，第二次得到什么？ |
| 解释 | 为什么 as_ref 不等于 &slot？ |
| 边界 | 若 boxed 在 take 之后仍会使用，能先取走 Box 吗？ |

先写下判断与原因，再用自己已有的代码验证；遇到疑问回查本篇，不需要提前读后续应用。

## 项目应用：阶段 8～9

ManagedTask 保存 Option<JoinHandle<T>>，take 转移已有句柄并留下 None。
先读[JoinHandle](../../libraries/tokio/join-handle.md)，再读[任务清理](../../walkthroughs/task-cleanup.md)。
