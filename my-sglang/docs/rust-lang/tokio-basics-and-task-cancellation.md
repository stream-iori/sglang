# Tokio 专题：异步基础、任务句柄与取消

[返回索引](index.md)

## 复习路线

```text
Future / async / await
        ↓
Runtime / 任务 / 线程
        ↓
spawn / async move / JoinHandle
        ↓
sleep / timeout / pending
        ↓
abort / Drop / 取消测试
```

## 1. 哪些属于 Rust，哪些属于 Tokio

| 概念 | 来自哪里 | 大白话 |
|---|---|---|
| `async`、`.await` | Rust 语言 | 描述异步工作、等待异步结果 |
| `Future`、`pending` | Rust 标准库 | 异步工作协议、永远不完成的 Future |
| Runtime、`spawn`、异步 `JoinHandle` | Tokio crate | 驱动异步任务、启动任务、等待或取消任务 |
| `sleep`、`timeout` | Tokio crate | 异步等待时间、限制等待时间 |
| Router、HTTP 提取器 | Axum crate | 接收 HTTP 请求并调用处理函数 |

Tokio 不会把整个程序自动变成非阻塞程序；放进异步函数的同步阻塞操作，仍然会阻塞线程。
见 [Tokio 官方教程](https://tokio.rs/tokio/tutorial)。

## 2. Future、async 与 await

| 写法 | 含义 | 容易误解的地方 |
|---|---|---|
| 调用 `async fn` | 得到一个 Future | 调用本身不等于已经执行完函数体 |
| `async { ... }` | 把这段工作描述为 Future | 不会自动创建后台任务 |
| `future.await` | 等待它产生结果 | 不会自动创建线程；若结果已就绪，也不必挂起 |

Future 被轮询时才能推进。尚未就绪时，它返回等待状态；等资源就绪，任务被唤醒后再继续。
普通异步 I/O 等待不会一直占着线程。不是所有 `.await` 都必然让出执行机会。
参见 [标准库 Future](https://doc.rust-lang.org/std/future/trait.Future.html)。

`.await` 也支持实现 `IntoFuture` 的值：不要把“能 await”机械理解为“这个类型直接实现了 Future”。
例如 Axum 0.8 的 `axum::serve(...)` 返回 `Serve`；等待它是在持续运行服务器，不是等一条请求。
普通 `serve(...).await` 不会在处理完一次请求后退出。参见 [Axum serve](https://docs.rs/axum/0.8.9/axum/fn.serve.html)。

## 3. Runtime、任务和线程

```text
操作系统线程
    └─ Tokio Runtime 调度任务
           ├─ 任务 A：等网络 → 暂停
           └─ 任务 B：有工作 → 继续运行
```

任务是 Runtime 管理的异步工作，不是一任务一线程。单线程 Runtime 也能让多项等待重叠；
多线程 Runtime 还能让任务在不同线程上运行。并发表示多项工作可以交错推进，
并行表示同一时刻真的在执行多项工作。参见 [Tokio spawning 教程](https://tokio.rs/tokio/tutorial/spawning)。

| 宏 | 做什么 | 默认 Runtime |
|---|---|---|
| `#[tokio::main]` | 创建 Runtime 并运行异步入口 | 多线程 |
| `#[tokio::test]` | 为异步测试创建 Runtime | 单线程；每个测试独立 |

默认值不是固定限制，可显式配置。见 [main 宏](https://docs.rs/tokio/1.53.1/tokio/attr.main.html)
和 [test 宏](https://docs.rs/tokio/1.53.1/tokio/attr.test.html)。

## 4. spawn、async move 与共享 Worker

### 4.1 六个概念分别负责什么

| 概念 | 职责 | 不要混淆 |
|---|---|---|
| `async { ... }` | 创建 Future，描述异步工作 | 创建时不执行块内的工作，也不创建线程 |
| `move` | 让 Future 按值捕获使用到的外部变量 | 不自动 clone，不指定运行线程 |
| `.await` | 推进并等待异步结果 | 等待未就绪时可让出执行机会，并非每次都暂停 |
| `tokio::spawn` | 将 Future 提交为独立调度的任务 | 任务不等于独占一个操作系统线程 |
| `Send + 'static` | 满足 spawn 的跨线程与借用有效性要求 | move 不会自动使 Future 满足这两个要求 |
| 取消与 `Drop` | 任务 Future 被销毁时清理其持有的值 | 丢弃 JoinHandle 不等于销毁正在运行的任务 |

捕获发生在创建 Future 时；块内创建 guard 等工作，要等 Future 被轮询后才执行。

### 4.2 不写 move，也可能移动变量

| 写法 | 捕获规则 |
|---|---|
| `async { ... }` | 根据块内用法，逐个推断共享借用、可变借用或按值捕获 |
| `async move { ... }` | 按值捕获块内使用到的外部变量 |

例如下面只读取 String，可以借用；当前函数直接等待，能保证借用期间 message 有效：

```rust
async fn print_message() {
    let message = String::from("hello");

    async {
        println!("{message}");
    }
    .await;
}
```

如果块内消费 String，编译器会推断移动，即使没有写 move：

```rust
let message = String::from("hello");
let future = async {
    drop(message);
};
// message 已移入 future，之后不能再使用。
```

`my-smg` 测试服务中的 `axum::serve(listener, app)` 同样按值取得 listener 和 app，
因此这段 async 块可以省略 move；保留 `async move` 则明确表达后台任务持有这些资源。
这不是“spawn 总能省略 move”，而是本次块内用法已经足以推断移动。

捕获规则见 [Rust Reference：async blocks 的捕获方式](https://doc.rust-lang.org/reference/expressions/block-expr.html#capture-modes)。

### 4.3 什么时候需要 move

只读取变量时，默认捕获通常是借用。交给 spawn 的任务可能比当前函数活得久，
不能依赖当前函数中局部 String 的借用，因此要把所有权交给 Future：

```rust
let message = String::from("hello");

let task = tokio::spawn(async move {
    println!("{message}");
});
```

| 场景 | 如何判断 |
|---|---|
| 当前函数直接 `.await`，外部变量一直有效 | 通常可以借用，不必强制 move |
| 后台任务使用外部拥有型数据 | 通常用 move，让任务独立拥有数据 |
| 块内本来就按值消费变量 | 不写 move 也可能推断为移动 |
| 调用方和后台任务都需要同一个 Worker | 先 Arc::clone，再将那份 Arc move 进去 |

move 对不同值的效果也不同：

| 捕获值 | 效果 |
|---|---|
| `String` 等非 Copy 值 | 移动所有权，外面的原变量不能继续使用 |
| `usize` 等 Copy 值 | 捕获副本，原变量仍能使用 |
| `Arc<Worker>` | 移动这份 Arc，不自动增加强引用计数 |
| `&Worker` | 捕获引用，不取得 Worker 的所有权，也不延长它的存活时间 |

move / Copy / 借用的基础见 [所有权、转移与借用](ownership-move-and-borrowing.md)。

### 4.4 Future 持有变量，为什么暂停后仍存在

可以把 Future 理解成保存执行进度的对象。编译器会让它保存恢复执行所需的捕获值、
跨越等待点的局部变量，以及正在等待的异步操作；这不是每暂停一次就重新创建变量。
具体布局由编译器决定，不需要手写状态机。

```text
创建 async move 块
    ↓
Future 持有捕获值：例如 Arc<Worker>
    ↓ 第一次推进
创建 guard：在途计数 +1
    ↓ 等待未就绪
Future 保存继续执行所需的 guard 等状态；线程可执行其他任务
    ↓
继续执行，或取消并销毁 Future
    ↓
guard Drop：在途计数 -1
```

所以普通在途 guard 可以跨越 await 保留，覆盖请求等待期间。它与持有锁的 MutexGuard
职责不同，见 [锁守卫与 await](mutex-guard-deref-and-poisoning.md#为什么不能持有-guard-跨越-await)。
Future 的保存状态机制见 [Rust Reference：async Future 的布局说明](https://doc.rust-lang.org/reference/expressions/block-expr.html#async-blocks)。

### 4.5 Arc、Send 与 static 的边界

`tokio::spawn(async move { ... })` 把工作提交给 Runtime，立即返回句柄。
“已经提交”不等于“函数返回时任务已经执行到创建 guard 那一行”。

`async move` 把捕获的值交给 Future，不是“移动到某个指定线程”。
取消测试用 `Arc::clone` 给任务一份所有权，仍然指向同一个 Worker：

```text
测试中的 worker ──Arc──┐
                     ├──> 同一个 Worker / 同一个计数器
任务中的 task_worker ─┘
```

`tokio::spawn` 要求 Future 满足 `Send + 'static`：任务可能在不同线程间移动，也可能比调用方
活得更久。这里 `'static` 不表示任务永远运行；表示不能依赖外部短期借用。
拥有一个 Arc 后，可以在任务内部再借用其中的 Worker。
见 [spawn 的生命周期和 Send 要求](https://tokio.rs/tokio/tutorial/spawning)。

`'static` 不要求任务永远运行，也不要求数据从程序启动就存在。拥有一个 String 的 Future
可以满足 `'static`，只要没有依赖可能失效的外部借用。move 一个短期引用进去，并不会修复它。

`Send` 需要检查 Future 的状态：例如在等待点仍保存的值能否跨线程转移。
如果捕获并跨 await 保存 `Rc`，即使用 async move，也不能提交给要求 Send 的 tokio::spawn。
详细要求见 [Tokio spawning 教程](https://tokio.rs/tokio/tutorial/spawning)。

### 4.6 结束、取消与释放

| 情况 | 持有的数据如何清理 |
|---|---|
| 正常执行离开局部作用域 | 相应局部值按作用域规则释放 |
| 未完成的 Future 被销毁 | 已捕获和已创建且仍保存的值被释放 |
| 普通异步任务收到 abort | 调度器完成取消后，销毁任务 Future；等待句柄确认清理 |
| 只丢弃 JoinHandle | 任务脱离句柄继续运行，不代表取消 |

尚未被轮询时，块内 guard 还没有创建，自然没有计数要减；已经创建后被取消，
才通过 Drop 清理。返回给调用方或移交给其他任务的值，由新的所有者负责后续释放。
取消不能自动撤销远端已执行的工作。完整过程见 [第 7 节](#7-abort-与-drop取消测试完整过程)。

## 5. JoinHandle：不是 Runnable

名字是 `JoinHandle`，不是 `JoinHandler`。它是已经提交的任务的控制句柄。

| Rust | Java 中用途较接近的概念 |
|---|---|
| 提交的异步工作 | `Runnable` / `Callable` 描述的工作 |
| `tokio::spawn(...)` | `ExecutorService.submit(...)` |
| Tokio `JoinHandle<T>` | 提交后返回的 `Future<T>` |

这是职责类比，不代表运行或取消机制相同；Rust 的 Future 也不等于 Java 的 Future。

| 类型 | 来源 | 等待方法 |
|---|---|---|
| `std::thread::JoinHandle<T>` | 标准库线程 | `.join()`，阻塞调用线程 |
| `tokio::task::JoinHandle<T>` | Tokio 异步任务 | `.await`，得到 `Result<T, JoinError>` |

任务不返回额外数据时，句柄是 `JoinHandle<()>`。正常完成得到 `Ok(())`，取消或 panic
得到 `JoinError`。`is_cancelled()` 用来区分取消。
**丢弃句柄会让任务脱离句柄继续运行，不等于取消。**
参见 [Tokio JoinHandle](https://docs.rs/tokio/1.53.1/tokio/task/struct.JoinHandle.html)。

## 6. sleep、timeout 和 pending

### sleep：等待时间，不占住线程

| 操作 | 影响 |
|---|---|
| `std::thread::sleep(...)` | 阻塞当前操作系统线程 |
| `tokio::time::sleep(...).await` | 当前任务等定时器，线程可以推进其他任务 |

`Duration` 是标准库里的时间长度类型；`from_secs(2)` 是 2 秒，`from_millis(1)` 是 1 毫秒。
异步定时器需要 Runtime 支持。见 [Tokio sleep](https://docs.rs/tokio/1.53.1/tokio/time/fn.sleep.html)。

### timeout：给一段异步等待设上限

本次等待“任务已经开始”的代码：

```rust
let started = tokio::time::timeout(Duration::from_secs(2), async {
    while worker.counter() == 0 {
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
})
.await;
```

```text
计数仍为 0 → 等 1 毫秒 → 再检查
计数变为 1 → 退出循环 → started = Ok(())
超过 2 秒  → 停止等待 → started = Err(Elapsed)
```

这个循环不是处理请求，而是确认 spawn 的任务确实创建了 guard。若直接 abort，任务可能
尚未执行，计数一直为 0，那么测试根本没测到 guard 的清理。

超时取消的是被包装的 Future，不会强行中断占住线程不让出的同步代码。
本例包住的是观察循环，**不是另一个已 spawn 的任务**；超时后仍需清理那个任务。
若把 JoinHandle 按值放进 timeout，超时丢弃句柄也不会自动 abort 任务。
见 [Tokio timeout](https://docs.rs/tokio/1.53.1/tokio/time/fn.timeout.html)。

### pending：故意让任务一直等

`std::future::pending::<()>()` 来自标准库，返回一个永远不会就绪的 Future。
`::<()>` 是 turbofish，指定预期输出类型为 unit；并不意味着它真的会返回这个值。
在取消测试里，它让任务停在“持有 guard、等待中”的状态，不会自己正常完成。
参见 [标准库 pending](https://doc.rust-lang.org/std/future/fn.pending.html)。

## 7. abort 与 Drop：取消测试完整过程

```text
测试函数                          被 spawn 的任务
创建 Arc<Worker>
spawn ──────────────────────────> 创建 guard：计数 +1
观察到计数 1                      pending.await：等待
abort ──────────────────────────> 发起取消
等待 JoinHandle                   Future 被销毁，guard Drop：计数 -1
确认 JoinError 是 cancelled
检查计数为 0
```

`abort()` 发起取消，不保证调用返回时任务已经结束；接着等待 `task.await`，才能确认清理完成。
对于正在进行的普通异步任务，取消会销毁它保存的局部值，因此 guard 的 Drop 能减一。
如果任务已正常完成，abort 也可能得到正常结果；本例用 pending 避免正常完成路径。
见 [任务取消规则](https://docs.rs/tokio/1.53.1/tokio/task/index.html)。

取消不是强制杀线程：不让出执行机会的 CPU 循环会拖延取消；已经运行的 `spawn_blocking`
任务也不能用 abort 停掉。取消更不代表数据库写入或远端请求会自动回滚。
见 [JoinHandle::abort](https://docs.rs/tokio/1.53.1/tokio/task/struct.JoinHandle.html#method.abort)。

这是本次教学测试的流程说明，不代表当前项目已经实现了全部场景。测试的意义是证明：
**任务没有走到正常结尾，计数仍能通过 Drop 归零。**

## 8. 与已有文档的联系

| 本专题问题 | 继续阅读 |
|---|---|
| Arc 克隆为何还是同一个 Worker | [Arc、Weak 与循环引用](arc-weak-and-cycle-references.md) |
| guard 借用为何不能比 Worker 活得更久 | [guard 的生命周期](ownership-move-and-borrowing.md#guard-借用-worker生命周期如何防止先销毁被借用者) |
| 计数如何通过共享引用修改 | [AtomicUsize 与内部可变性](atomic-usize-and-interior-mutability.md) |
| 策略的 MutexGuard 为何要在 await 前释放 | [锁与 await](mutex-guard-deref-and-poisoning.md#为什么不能持有-guard-跨越-await) |
| `pending::<()>()` 的尖括号和括号是什么 | [pending 的语法拆解](borrowed-slices-and-vec-type-annotation.md#pending泛型类型与函数调用) |
| `Result`、`expect_err` 如何理解 | [Result 与错误传播](result-question-mark-and-error-propagation.md) |

普通 `InFlightGuard` 不持有锁；它跨越 await 是为了覆盖请求等待期间。
标准库 `MutexGuard` 则持有锁，作用和约束不同，不能只因名字都含 guard 就混为一谈。

## 一句话复习

```text
async 描述工作，Runtime 驱动工作，spawn 提交工作，JoinHandle 观察工作。
await 不等于阻塞线程，spawn 不等于创建线程，丢弃句柄不等于取消任务。
取消后等待任务结束，再检查 guard 是否把计数清理干净。
```
