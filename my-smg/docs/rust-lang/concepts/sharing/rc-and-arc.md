# Rc 与 Arc：多个句柄共同拥有数据

[分类目录](index.md) · [总入口](../../index.md) · [问题索引](../../questions.md)

**只要还有强引用句柄，数据就活着；最后一个强引用释放时销毁数据。**

| 阅读准备 | 内容 |
|---|---|
| 前置 | [共享借用与可变借用](../ownership/borrowing.md) · [Drop 与 RAII：离开作用域如何收尾](../ownership/drop-and-raii.md) |
| 源码入口 | [src/worker_registry.rs](../../../../src/worker_registry.rs)：`WorkerRegistry::get` |
| 阅读时机 | 按当前课程需要阅读，不据此跳过 learning 的阶段验收。 |

## 先不谈线程：引用与拥有句柄不同

独立语法例子：

```rust
use std::rc::Rc;
let a = Rc::new(100);
let b = Rc::clone(&a);
drop(a);
assert_eq!(*b, 100);
```

```text
a：Rc ──┐
        ├──> 同一份 100
b：Rc ──┘

a 释放 → 强引用从 2 到 1 → 数据仍在
b 释放 → 强引用从 1 到 0 → Rc 自动销毁数据
```

`b = &a` 只是借用 a，不能在 a 已销毁后继续使用；`Rc::clone(&a)` 创建另一个共享拥有句柄，可以独立于 a 这个绑定继续存活。

| 类型 | 解决什么 | 限制 |
|---|---|---|
| `&T` | 临时共享访问 | 外部拥有者必须保证数据存活 |
| `Rc<T>` | 单线程中共同拥有 | 非原子计数，不能跨线程转移共享句柄 |
| `Arc<T>` | 使用原子计数共同拥有 | 跨线程使用还要求内层类型满足相应 Send/Sync 约束 |

项目 `WorkerRegistry::get` 克隆 Arc，让调用者和注册表共同拥有同一个 Worker。Arc 保证引用计数安全，不自动允许修改任意内层数据。

## `Arc` 是共享所有权，不是绕过所有权

多个异步任务都需要长期持有同一个不可变配置时，可以使用 `Arc<T>`：

```rust
use std::sync::Arc;

let config = Arc::new(String::from("production"));

let first = Arc::clone(&config);
let second = Arc::clone(&config);

assert!(Arc::ptr_eq(&first, &second));
```

```text
config ──┐
first  ──┼──> 同一个 String
second ──┘
```

每个 `Arc` 都是一个所有者；最后一个 `Arc` 被 drop 时，内部值才释放。`Arc<T>` 只提供共享
所有权，不会自动允许修改 `T`。共享可变状态还需要 `Mutex`、`RwLock` 等同步机制，并需要单独
设计并发语义。
上面的 `Arc::ptr_eq` 用于确认两份 Arc 指向同一个对象，区别于值相等；详见
[Arc：对象身份与值相等](../../walkthroughs/circuit-permit.md)。

`Arc`、`Weak` 的生命周期关系见 [`Arc`、`Weak` 与循环引用](weak-and-cycles.md)。
`Arc<Mutex<T>>` 的共享修改、锁守卫解引用与自动解锁见
[`Mutex`、`MutexGuard`、解引用与毒锁](mutex-and-guards.md)。

## `Arc<dyn Policy>`：共享所有权

`Arc<T>` 使用线程安全的原子引用计数，让多个所有者指向同一个值：

```rust
let policy: Arc<dyn ReadOnlyPolicy + Send + Sync> =
    Arc::new(SomeConcretePolicy::new());

let another_owner = Arc::clone(&policy);
```

`Arc::clone` 只增加引用计数，不会复制具体策略。最后一个强引用释放时，策略才会被销毁。

但是：

```text
Arc 解决共享所有权
Arc 不自动解决共享可变性
```

当前 `Policy::select` 要求 `&mut self`。多个线程同时持有 `Arc<dyn Policy>` 时，不能直接从共享
`Arc` 获得 `&mut dyn Policy`。可能的后续设计包括：

- 把策略状态改成 `AtomicUsize` 等内部可变且并发安全的类型；
- 使用 `Arc<Mutex<Box<dyn Policy + Send>>>` 串行化可变访问；
- 重新设计 trait，让方法只需要 `&self`，并由具体实现负责同步。

这正是上游轮询策略使用 `AtomicUsize` 后能把方法写成 `select_worker(&self, ...)` 的原因。
原子操作为何能使用 `&self`、`Ordering::Relaxed` 的含义见
[`AtomicUsize` 与内部可变性](atomics.md)。

`Arc`、强弱引用计数与循环引用详见
[`Arc`、`Weak` 与循环引用](weak-and-cycles.md)。

## `Arc` 中的值不能可变借用

`Arc<T>` 只允许多个所有者共享访问。若行为需要修改状态，应选择原子类型、锁或新的状态边界，
不能把 `Arc` 当成自动提供 `&mut T` 的工具。

## 判断该用哪个

| 关系 | 建议 | 原因 |
|---|---|---|
| 父拥有子 | `Arc<Child>` | 父存在时子必须存在 |
| 子回指父 | `Weak<Parent>` | 子不应该延长父的生命周期 |
| 缓存 / registry 中的可失效条目 | 常用 `Weak<T>` | 缓存不应成为对象唯一所有者 |
| 独立且必须共同存活的共享对象 | `Arc<T>` | 调用方需要明确拥有它 |

`Arc<T>` 只解决共享所有权。如果多个任务还要修改同一份 `T`，通常需要组合成
`Arc<Mutex<T>>`；Guard 解引用、毒锁和异步持锁边界见
[`Mutex`、`MutexGuard`、解引用与毒锁](mutex-and-guards.md)。

## 自查与关联

Arc::clone 是否创建新的 Worker？

[Box：独占拥有堆上对象](box.md) · [Weak 与强引用环](weak-and-cycles.md) · [RefCell：单线程里的运行时借用检查](refcell.md)
