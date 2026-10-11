# Rust：`Arc`、`Weak` 与循环引用

## 结论

`Weak<T>` 用于“能找到对象，但不拥有对象”。它不增加强引用计数，因此把循环中的至少一条边改成
`Weak` 后，对象就能正常释放。

```rust
let weak_context = Arc::downgrade(&app_context);
```

这句把：

```text
app_context: Arc<AppContext>
        │
        │ 借用，不转移所有权
        ▼
Arc::downgrade(&app_context)
        │
        ▼
weak_context: Weak<AppContext>
```

## `Arc` 与 `Weak` 的区别

| 类型 | 是否拥有对象 | 是否增加强引用计数 | 能否保证对象存活 |
|---|---:|---:|---:|
| `Arc<T>` | 是 | 是 | 是 |
| `Weak<T>` | 否 | 否 | 否 |

可以把强引用计数理解成对象的“存活票数”：

```text
strong_count > 0  -> T 一定还活着
strong_count == 0 -> T 被销毁
```

`Weak<T>` 不会让 `strong_count` 增加，所以它不能单独让 `T` 继续存活。

## ptr_eq：对象身份与值相等

`Arc::ptr_eq(&first, &second)` 返回 `bool`，检查两份 Arc 是否指向**同一份分配、同一个对象**，
而不是检查对象内容是否相等。它借用两份 Arc，不移动所有权，也不增加强引用计数。
参见 [标准库 Arc::ptr_eq](https://doc.rust-lang.org/std/sync/struct.Arc.html#method.ptr_eq)。

```rust
use std::sync::Arc;

let original = Arc::new(String::from("hello"));
let cloned = Arc::clone(&original);
let same_value = Arc::new(String::from("hello"));

assert!(Arc::ptr_eq(&original, &cloned));
assert!(!Arc::ptr_eq(&original, &same_value));
assert_eq!(original, same_value); // 内容相同，但不是同一个对象
```

```text
original ──┐
cloned ────┴──> String("hello")，对象 A
same_value ───> String("hello")，对象 B
```

| 检查 | 问的问题 | 需要内部类型实现 PartialEq 吗 |
|---|---|---|
| `Arc::ptr_eq(&a, &b)` | 是不是同一个对象 | 不需要 |
| `a == b` / `assert_eq!(a, b)` | 内部值按 PartialEq 规则是否相等 | 需要 |

在节点注册表的“拒绝重复 ID”测试里，原节点和重复节点可以有相同 ID，但它们是两个对象。
检查 `Arc::ptr_eq(&original, &found)`，才能确认 get 返回的仍是原对象，而不是被偷偷替换成
另一个同名节点。只检查 ID 相等，发现不了这种替换。

## 节点移除与对象销毁是两件事

从注册表移除节点，是删除注册表持有的条目；销毁 Worker，是释放最后一个拥有者之后才发生的事。
`Arc::clone` 不复制 Worker，只给同一个 Worker 增加一个强引用所有者。

下面只画注册表和一个请求，实际还可能有其他所有者：

```text
移除前：
注册表 ──Arc──┐
             ├──> 同一个 Worker 对象
请求 guard ──┘

移除后：
注册表：不再有这个条目
请求 guard ──Arc──> Worker 仍然存活

请求结束：
guard 减计数并释放 Arc
若没有其他强引用 → Worker 才销毁
```

因此新查询可以看不到这个节点，已有请求仍能访问它。这里保住的是 **Worker 对象**，
不是把已经删除的注册表条目重新放回去。

### 如果 guard 只保存引用

| guard 保存什么 | 能否独立保住对象 | 当前 Vec 场景下移除会怎样 |
|---|---|---|
| `&Worker` | 不能；引用不是拥有者 | 移除破坏仍有效的借用时，编译器拒绝代码 |
| `Arc<Worker>` | 能；guard 是共同拥有者 | 可从列表移除，对象由剩余 Arc 保持存活 |

对于当前 `my-smg` 的 `Vec<Worker>`，下面的顺序不能编译：

```rust,compile_fail
let guard = workers[0].begin_request();
workers.remove(0); // 需要可变访问列表，但 guard 仍借用其中的节点
drop(guard);
```

“编译器保证”不是运行时自动等待：它在编译时拒绝危险顺序。先销毁 guard，再移除节点，
才符合这个借用设计。并非任何引用都禁止注册表修改；如果先独立持有 Arc，再从它借用，
对象存活的根本保证仍来自那份 Arc。

### 原项目为什么选择 Arc

| 源码事实 | 设计效果 |
|---|---|
| [WorkerRegistry::get](../../../sgl-model-gateway/src/core/worker_registry.rs#L373) 返回克隆的 Arc | 查询结束后不必继续持有注册表条目的借用或锁 |
| [WorkerRegistry::remove](../../../sgl-model-gateway/src/core/worker_registry.rs#L316) 移除条目与索引，并标记不健康 | 更新后续查询和选择所看到的节点资格，而不是立即销毁所有请求中的对象 |
| [WorkerLoadGuard](../../../sgl-model-gateway/src/core/worker.rs#L1147) 拥有 `Arc<dyn Worker>` | 即使注册表移除节点，guard 仍能在结束时减计数 |
| guard 可随响应体存活 | 流式响应持续时，不依赖处理函数的局部借用 |

这不是“Arc 比引用更高级”，而是需要请求的所有权独立于注册表借用。
`dyn Worker` 负责具体实现的抽象，`Arc` 负责共同拥有，两个问题不要混为一谈。
`Arc` 也不会自动同步可变字段、停止远端进程或撤销已经发出的请求。

设计感悟见 [realization：可选资格与对象存活分离](realization.md)。
引用与生命周期的编译期约束见
[guard 借用 Worker](ownership-move-and-borrowing.md#guard-借用-worker生命周期如何防止先销毁被借用者)。

## 为什么两个 `Arc` 会泄漏

假设父子双方都用 `Arc` 保存对方：

```text
外部 ──Arc──> Parent ──Arc──> Child
                  ▲                 │
                  └─────Arc─────────┘
```

当外部引用释放后：

```text
Parent strong_count = 1  （Child 持有）
Child  strong_count = 1  （Parent 持有）
```

两个对象都在等待对方先释放，但谁也不会先释放：

```text
Parent 释放条件：strong_count == 0  ✗
Child  释放条件：strong_count == 0  ✗
```

结果是内存泄漏。

## 用 `Weak` 断开“反向指针”

通常让“拥有关系”保持 `Arc`，让“回指 / 父指针 / 注册表反查”变成 `Weak`：

```text
外部 ──Arc──> Parent ──Arc──> Child
                  ▲                 │
                  └────Weak─────────┘
```

外部释放时，释放顺序变成：

```text
1. 外部 Arc<Parent> 被 drop
2. Parent strong_count = 0 -> Parent 被释放
3. Parent 持有的 Arc<Child> 被 drop
4. Child strong_count = 0 -> Child 被释放
5. Child 中的 Weak<Parent> 自动失效
```

关键不是“`Weak` 自动清理循环”，而是：

```text
Weak 不算拥有者
=> 它不会阻止 strong_count 降到 0
=> 对象可以析构
```

## `upgrade()`：弱引用不能直接使用

`Weak<T>` 指向的对象可能已经释放。使用前必须升级：

```rust
if let Some(context) = weak_context.upgrade() {
    // context: Arc<AppContext>
    // AppContext 仍存活；在此作用域内可安全使用。
} else {
    // 没有任何 Arc<AppContext> 了，对象已经释放。
}
```

类型变化：

```text
Weak<AppContext>
       │ .upgrade()
       ▼
Option<Arc<AppContext>>
       │
       ├─ Some(Arc<_>)：对象存活
       └─ None        ：对象已释放
```

不能把 `Weak` 当作永远有效的 `Arc`：

```rust
// ❌ weak_context.do_something()

// ✅ 先升级
let context = weak_context.upgrade()?;
context.do_something();
```

## SGL Model Gateway 中的例子

SMG 启动时创建 `AppContext` 后，创建 JobQueue：

```rust
let app_context = Arc::new(...);
let weak_context = Arc::downgrade(&app_context);
let worker_job_queue = JobQueue::new(JobQueueConfig::default(), weak_context);
```

源码见 [`server.rs:810-823`](../../../sgl-model-gateway/src/server.rs#L810-L823)。关系是：

```text
AppContext ──Arc──> JobQueue
     ▲                  │
     └──────Weak────────┘
```

如果 JobQueue 反过来持有 `Arc<AppContext>`，就会形成强引用环：

```text
AppContext ──Arc──> JobQueue
     ▲                  │
     └──────Arc─────────┘  <- 无法释放
```

用 `Weak<AppContext>` 后，关闭服务并释放最后一个 `Arc<AppContext>` 时，JobQueue 不会阻止
AppContext 析构。

## 最小可运行模型

```rust
use std::sync::Arc;

let strong = Arc::new(String::from("hello"));
let weak = Arc::downgrade(&strong);

assert_eq!(Arc::strong_count(&strong), 1);
assert_eq!(Arc::weak_count(&strong), 1);

assert_eq!(weak.upgrade().as_deref(), Some("hello"));

drop(strong);

assert!(weak.upgrade().is_none());
```

`drop(strong)` 后，字符串已被释放；变量 `weak` 仍然存在，但它只能返回 `None`。

## 判断该用哪个

| 关系 | 建议 | 原因 |
|---|---|---|
| 父拥有子 | `Arc<Child>` | 父存在时子必须存在 |
| 子回指父 | `Weak<Parent>` | 子不应该延长父的生命周期 |
| 缓存 / registry 中的可失效条目 | 常用 `Weak<T>` | 缓存不应成为对象唯一所有者 |
| 独立且必须共同存活的共享对象 | `Arc<T>` | 调用方需要明确拥有它 |

`Arc<T>` 只解决共享所有权。如果多个任务还要修改同一份 `T`，通常需要组合成
`Arc<Mutex<T>>`；Guard 解引用、毒锁和异步持锁边界见
[`Mutex`、`MutexGuard`、解引用与毒锁](mutex-guard-deref-and-poisoning.md)。

## 一句话记忆

```text
Arc：我拥有你，你不能先消失。
Weak：我认识你；你还在我才用你。
```
