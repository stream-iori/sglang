# Rust：`Mutex`、`MutexGuard`、解引用与毒锁

## 结论

```text
Arc       解决：多个任务如何拥有同一份状态
Mutex     解决：同一时刻只允许一个任务修改状态
MutexGuard解决：锁保持多久，以及如何访问锁内数据
```

`my-smg` 的失败计数器可以表示为：

```rust
Arc<Mutex<usize>>
```

完整访问链：

```text
Arc<Mutex<usize>>
        │ .lock()
        v
Result<MutexGuard<'_, usize>, PoisonError<_>>
        │ match
        v
MutexGuard<'_, usize>
        │ * 解引用
        v
usize
```

## 为什么需要 `Arc<Mutex<T>>`

多个 Axum 请求会拿到克隆后的 `AppState`。如果状态中保存：

```rust
failures_remaining: Arc<Mutex<usize>>
```

克隆 `AppState` 时，只会克隆 `Arc`；所有副本仍然指向同一个 `Mutex<usize>`：

```text
请求 A ──Arc──┐
请求 B ──Arc──┼──> Mutex<usize>
请求 C ──Arc──┘
```

两层类型职责不同：

| 类型 | 解决的问题 |
|---|---|
| `Arc<T>` | 多个任务共同拥有同一个 `T` |
| `Mutex<T>` | 对 `T` 的可变访问必须串行进行 |

`Arc<T>` 本身不允许多个任务安全地同时修改 `T`。`Mutex<T>` 也不负责多个长期所有者，因此并发
服务中经常组合成 `Arc<Mutex<T>>`。

`Arc` 的所有权与引用计数见 [`Arc`、`Weak` 与循环引用](arc-weak-and-cycle-references.md)。
当 Mutex 保护的是需要保存轮询进度的 `Box<dyn Policy + Send>` 时，见
[配置驱动的共享策略](traits-dispatch-and-smart-pointers.md#my-smg配置驱动的共享策略)。

## `lock()` 为什么返回 `Result`

```rust
state.failures_remaining.lock()
```

返回类型可以展开为：

```rust
Result<
    MutexGuard<'_, usize>,
    PoisonError<MutexGuard<'_, usize>>,
>
```

也可以写成标准库的类型别名：

```rust
LockResult<MutexGuard<'_, usize>>
```

两个分支是：

```text
Ok(MutexGuard)          正常取得锁
Err(PoisonError)        取得锁，但 Mutex 已被标记为中毒
```

因此需要从 `Result` 中取出 Guard：

```rust
let mut remaining = match state.failures_remaining.lock() {
    Ok(guard) => guard,
    Err(poisoned) => poisoned.into_inner(),
};
```

这里的 `guard` 并没有被忽略。`match` 是表达式，每个分支都会产生最终赋给 `remaining` 的值：

```text
Ok 分支  ──> guard ───────────────┐
                                  ├─> remaining: MutexGuard<'_, usize>
Err 分支 ──> poisoned.into_inner()┘
```

这与 [`Result`、`?` 与错误传播](result-question-mark-and-error-propagation.md) 中的普通
`Result<T, E>` 是同一个模型，只是这里的 `T` 是 `MutexGuard`。

## 什么是毒锁

线程持有 `std::sync::Mutex` 时发生 panic，Mutex 会被标记为 poisoned：

```text
线程取得 MutexGuard
        │
        v
修改共享数据时 panic
        │
        v
MutexGuard 被 drop，锁被释放
        │
        v
Mutex 被标记为 poisoned
```

“中毒”不表示锁永远无法打开，而是在提醒调用者：

> 上一次修改可能只做了一半，内部数据的一致性需要重新判断。

下一次 `lock()` 返回 `Err(PoisonError)`。错误内部仍然携带取得的 Guard：

```rust
Err(poisoned) => poisoned.into_inner()
```

`into_inner()` 会消耗 `PoisonError`，取出其中的 `MutexGuard`。

```text
PoisonError<MutexGuard<T>>
           │ into_inner()
           v
      MutexGuard<T>
```

是否应该恢复取决于数据：

| 数据 | 常见选择 |
|---|---|
| 简单计数器 | 可以检查后继续使用 |
| 多字段且存在一致性约束的业务状态 | 先校验或重建，不能盲目继续 |

所以 `into_inner()` 不是“毒锁一定可以忽略”，而是显式选择继续检查或使用内部数据。

## `MutexGuard<'_, usize>` 的两个参数

```text
MutexGuard<'生命周期, 数据类型>
```

在：

```rust
MutexGuard<'_, usize>
```

| 参数 | 种类 | 含义 |
|---|---|---|
| `'_` | 生命周期参数 | 由编译器推断 Guard 对 Mutex 的借用时间 |
| `usize` | 类型参数 | Mutex 内部保护的数据类型 |

`'_` 不是类型，也不是 `'static`。它表达的约束是：

```text
Mutex 存活并被借用
        │
        └─ MutexGuard 不能比这个 Mutex 活得更久
```

生命周期的通用概念见 [`Formatter<'_>`、生命周期、`Display` 与 `Error`](formatter-lifetimes-display-and-error.md)。

## `MutexGuard` 为什么能用 `*` 解引用

`MutexGuard<T>` 是 RAII 锁守卫，同时具有智能指针式访问能力。它实现了：

```text
Deref<Target = T> 读取内部 T
DerefMut           修改内部 T
Drop               离开作用域时释放锁
```

因此：

```rust
remaining: MutexGuard<'_, usize>
*remaining: usize
```

可以画成：

```text
remaining
MutexGuard<usize>
        │ *
        v
锁保护的 usize
```

这与 `Box<T>` 的解引用形式相似：

```rust
let boxed = Box::new(10);
let value = *boxed;
```

区别是 `MutexGuard` 还代表锁的持有状态。

### 读取

```rust
if *remaining == 0 {
    // 比较内部 usize
}
```

不能直接写 `remaining == 0`，因为那是在比较 `MutexGuard<usize>` 和整数。

### 修改

```rust
*remaining -= 1;
```

可以近似理解为：

```rust
*remaining = *remaining - 1;
```

这需要两个条件：

```text
MutexGuard 实现 DerefMut
remaining 绑定声明为 mut
```

所以代码必须是：

```rust
let mut remaining = /* MutexGuard */;
*remaining -= 1;
```

`mut` 不是为了替换 Guard，而是为了通过 Guard 取得内部数据的可变访问。

### 解引用不一定意味着移动数据

`usize` 实现 `Copy`，因此读取会复制数值：

```rust
let count = *remaining;
```

如果 Mutex 内部是不可 `Copy` 的 `String`，不能直接把值从 Guard 后面移动出来：

```rust
// let text = *guard;       // 尝试移动 String，通常无法编译
let text_ref = &*guard;     // 借用 String
let text = (*guard).clone(); // 明确复制 String
```

所有权、`Copy` 和移动的区别见[所有权、转移与借用](ownership-move-and-borrowing.md)。

## 为什么用额外的大括号

失败计数代码写成：

```rust
let should_fail = {
    let mut remaining = match state.failures_remaining.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };

    if *remaining == 0 {
        false
    } else {
        *remaining -= 1;
        true
    }
};

tokio::time::sleep(state.delay).await;
```

大括号同时完成两件事：

```text
内部 block 的最后一个表达式
        │
        └─ 生成 should_fail: bool

离开内部 block
        │
        └─ remaining 被 drop，Mutex 自动解锁
```

这是 RAII：资源的释放绑定到值的生命周期，而不是手动调用 `unlock()`。

## 为什么不能持有 Guard 跨越 `.await`

错误的时间线：

```text
请求 A：lock，取得 Guard
        │
        ├─ sleep(...).await：A 暂停，但 Guard 仍存在
        │
请求 B：调用 std::sync::Mutex::lock()
        │
        └─ 锁仍被 A 占用，B 阻塞 Tokio 工作线程
```

`sleep().await` 本身不阻塞线程。真正的问题是：A 暂停时仍占着锁，其他任务调用标准库的阻塞
`lock()` 后会占住 Tokio 工作线程。

如果多个工作线程都阻塞在同一把锁上，负责唤醒 A 的异步任务可能无法及时继续，导致吞吐下降、
任务饥饿，严重时形成类似死锁的停滞。

另外，`std::sync::MutexGuard` 不能安全地在线程之间移动。Guard 跨越 `.await` 时，handler 生成的
Future 往往不满足 Axum 所需的 `Send`，代码可能直接无法编译。

正确边界是：

```text
lock
  v
只做快速的同步读写
  v
drop Guard，释放锁
  v
网络、sleep 等 .await
```

`tokio::sync::Mutex` 允许 Guard 跨越 `.await`，但“允许”不代表应该长时间持锁。只保护普通内存
计数器且临界区很短时，可以使用 `std::sync::Mutex`，前提是必须在 `.await` 前释放 Guard。

## 与模式解构、智能指针的联系

| 当前知识 | 相关概念 | 文档 |
|---|---|---|
| `Ok(guard) => guard` | `match` 模式和表达式返回值 | [模式解构、`if let`、`ref` 与解引用](if-let-ref-and-deref.md) |
| `*remaining` | `Deref`、借用与移动 | [所有权、转移与借用](ownership-move-and-borrowing.md) |
| `Arc<Mutex<T>>` | 共享所有权与共享可变状态 | [`Arc`、`Weak` 与循环引用](arc-weak-and-cycle-references.md) |
| `lock() -> Result` | 错误分支和传播 | [`Result`、`?` 与错误传播](result-question-mark-and-error-propagation.md) |
| `MutexGuard<'_, T>` | 推断生命周期 | [`Formatter<'_>` 与生命周期](formatter-lifetimes-display-and-error.md) |

## 一句话记忆

```text
Arc 让大家找到同一把锁。
Mutex 保证同一时刻只有一个修改者。
MutexGuard 既代表持锁，也像智能指针一样访问内部数据。
*guard 访问锁内值；guard drop 自动解锁。
毒锁是数据可能不一致的警告，不是锁永久失效。
锁内只做短操作，任何 .await 都放到锁外。
```
