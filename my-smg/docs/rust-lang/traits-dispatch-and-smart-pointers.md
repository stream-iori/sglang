# Rust：Trait、静态分发、动态分发与智能指针

## 一句话

```text
trait         定义行为契约
P: Trait      编译期确定具体类型，静态分发
dyn Trait     运行时通过虚表调用，动态分发
&dyn Trait    临时借用一个 trait object
Box<dyn Trait> 独占拥有一个类型被擦除的实现
Arc<dyn Trait> 多处共享拥有一个类型被擦除的实现
```

`static dispatch` 中的 `static` 指“编译期确定调用目标”，不是 Java 的 `static` 方法。

## 从两个策略开始

假设模型网关有两种选择策略：

```rust
pub struct FirstHealthy;

pub struct RoundRobin {
    next: usize,
}
```

它们内部状态不同：

- `FirstHealthy` 没有字段，每次选择第一个健康节点；
- `RoundRobin` 保存 `next`，每次选择后修改计数器。

调用方真正关心的是共同能力：给定节点列表，返回一个可用节点的原始索引。

```rust
pub trait Policy {
    fn select(&mut self, workers: &[Worker]) -> Option<usize>;
}
```

这就是行为契约。它规定：

- 调用需要策略的可变借用 `&mut self`；
- 节点列表只被共享借用；
- 成功时返回原始节点索引；
- 没有可用节点时返回 `None`。

trait 不规定实现是否使用计数器、哈希、随机数或请求信息。

## Trait 与 Java interface 的类比

可以先把 Rust trait 类比为 Java interface：类型通过实现共同接口，让调用方依赖契约而不是
具体实现。

```rust
impl Policy for FirstHealthy {
    fn select(&mut self, workers: &[Worker]) -> Option<usize> {
        // FirstHealthy 的实现
    }
}

impl Policy for RoundRobin {
    fn select(&mut self, workers: &[Worker]) -> Option<usize> {
        // RoundRobin 的实现
    }
}
```

但两者不能完全等同：

| 方面 | Rust trait | Java interface |
|---|---|---|
| 常见分发方式 | 泛型静态分发或 `dyn` 动态分发 | 对象方法通常动态分发 |
| 实现位置 | 可为已有类型实现 trait，但受 orphan rule 限制 | 类声明 `implements` |
| 数据大小 | `dyn Trait` 本身是动态大小类型 | 对象变量本来就是引用 |
| 组合约束 | `T: A + B`、`dyn A + Send + Sync` | `T extends A & B` 等 |
| 关联能力 | 关联类型、关联常量、泛型方法 | 泛型与默认方法等机制不同 |

当前阶段最重要的差别是：Rust 只有显式写出 `dyn Policy` 时，才明确要求 trait object 的动态
分发。

## 为什么共同签名使用 `&mut self`

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

所有权与借用基础见[所有权、转移与借用](ownership-move-and-borrowing.md)。

<a id="generic-struct-and-impl"></a>

## 泛型结构体与泛型 impl

泛型不只用于函数，也可以表示“结构相同、内部数据类型不同”的结构体。
`my-smg` 测试中的定义是：

```rust
struct ManagedTask<T> {
    handle: Option<JoinHandle<T>>,
}
```

这里 `T` 是任务成功时的返回值类型，不是线程类型，也不是必然指向 Worker。

| 具体类型 | 管理的任务 |
|---|---|
| `ManagedTask<()>` | 不返回额外数据的服务器或计数测试任务 |
| `ManagedTask<(StatusCode, String)>` | 返回状态码和正文的 chat 任务 |

```rust
impl<T> ManagedTask<T> { /* 为所有 T 提供固有方法 */ }
impl<T> Drop for ManagedTask<T> { /* 为所有 T 实现 Drop trait */ }
```

| 部分 | 作用 |
|---|---|
| `impl<T>` | 声明本次实现使用的类型参数 T |
| `ManagedTask<T>` | 指定为哪一组具体类型提供实现 |
| `Drop for` | 实现 trait，而不是定义普通固有方法 |

类型参数名字不必叫 T；没有写 bound，也不表示它自动具有 `Clone`、`Send` 等全部能力。
调用 `ManagedTask::new(handle)` 时，编译器可以从 `JoinHandle<T>` 推断 T，不必手写 turbofish。
泛型本身不要求堆分配或动态分发。

关联阅读：[接收者与 take](ownership-move-and-borrowing.md#consuming-self-and-option-take)、
[闭包参数中的 impl Trait](closures-and-fn-traits.md#fn-family)、
[Tokio 资源管理实例](tokio-basics-and-task-cancellation.md#managed-task-resource-lifecycle)。

## 泛型：静态分发

统一调用入口可以写成泛型函数：

```rust
pub fn select_with_policy<P: Policy>(
    policy: &mut P,
    workers: &[Worker],
) -> Option<usize> {
    policy.select(workers)
}
```

`P: Policy` 是 trait bound，表示 `P` 必须实现 `Policy`。

它更接近 Java 的泛型工具方法：

```java
static <P extends Policy> Integer selectWithPolicy(
    P policy,
    List<Worker> workers
) {
    return policy.select(workers);
}
```

它不是工厂方法，因为它没有创建策略，只接收已有策略并调用其行为。

### 编译器如何处理泛型调用

调用两种具体类型时：

```rust
select_with_policy(&mut first_healthy, &workers);
select_with_policy(&mut round_robin, &workers);
```

可以近似理解为编译器分别生成两个具体版本：

```text
select_with_policy::<FirstHealthy>(...)
select_with_policy::<RoundRobin>(...)
```

这个过程通常称为单态化（monomorphization）。具体调用目标在编译期已知，因此叫静态分发。

### 静态分发的特点

| 特点 | 影响 |
|---|---|
| 编译期知道具体类型 | 容易内联和优化 |
| 不需要虚表查找 | 调用开销通常很小 |
| 每种具体类型可能生成一份代码 | 泛型实例较多时会增加编译时间和二进制体积 |
| 局部变量仍有唯一具体类型 | 同一个普通变量不能一会儿是 `FirstHealthy`、一会儿是 `RoundRobin` |

最后一点很关键：泛型表达“可接受多种类型”，不代表一个变量能在运行时随意改变具体类型。

## `impl Trait` 与静态分发

参数位置的 `impl Policy` 是泛型参数的简写之一：

```rust
fn select_with_policy(
    policy: &mut impl Policy,
    workers: &[Worker],
) -> Option<usize> {
    policy.select(workers)
}
```

它仍然是静态分发，不等于 `dyn Policy`。

返回位置的 `impl Policy` 表示函数隐藏一个确定的具体返回类型：

```rust
fn default_policy() -> impl Policy {
    FirstHealthy::new()
}
```

函数的所有返回路径仍必须解析为同一种具体类型。若要根据运行时配置返回不同具体策略，通常
改用 `Box<dyn Policy>`。

## `dyn Trait`：动态分发

动态入口可以接收 trait object：

```rust
pub fn select_with_dynamic_policy(
    policy: &mut dyn Policy,
    workers: &[Worker],
) -> Option<usize> {
    policy.select(workers)
}
```

这里调用方只知道“它实现了 `Policy`”，具体是 `FirstHealthy` 还是 `RoundRobin` 要到运行时从
trait object 的虚表中确定。

概念上，trait object 的指针包含两部分：

```text
&mut dyn Policy
├─ data pointer   -> 具体策略值
└─ vtable pointer -> 该具体类型的 Policy 方法表及相关元数据
```

调用 `policy.select(...)` 时，会通过虚表找到当前具体类型的实现。这就是动态分发。

这是概念模型；代码不应依赖 trait object 的具体内存布局细节。

## 为什么通常不能单独写 `dyn Policy`

不同实现的大小可能不同：

```text
FirstHealthy：没有字段
RoundRobin：包含 usize 计数器
其他策略：可能包含 Vec、HashMap 或配置
```

编译器无法为一个裸 `dyn Policy` 局部值确定固定栈大小。因此 `dyn Trait` 属于动态大小类型，
通常必须放在某种指针后面：

```rust
&dyn Policy
&mut dyn Policy
Box<dyn Policy>
Arc<dyn Policy>
```

这些指针自身大小固定，并携带访问具体值和虚表所需的信息。

## `&dyn Policy`：只借用，不拥有

如果 trait 方法只需要 `&self`，可以使用：

```rust
fn inspect(policy: &dyn Policy) {
    // 共享借用 trait object
}
```

当前 `Policy::select` 需要 `&mut self`，因此调用入口使用：

```rust
fn select(policy: &mut dyn Policy, workers: &[Worker]) {
    // 独占可变借用 trait object
}
```

借用形式的特点：

- 不取得具体策略的所有权；
- 一般不因为类型擦除而要求堆分配；
- 生命周期不能超过原策略；
- 调用结束后，原所有者继续拥有策略。

引用是借用，不是拥有型智能指针。这里把它列出，是因为 trait object 必须通过某种指针形式
使用，而引用是最轻量的选择。

## `Box<dyn Policy>`：独占所有权与运行时切换

`Box<T>` 在堆上保存一个 `T`，并独占拥有它。`Box` 离开作用域时，内部值会被释放。

```rust
let mut policy: Box<dyn Policy> = Box::new(FirstHealthy::new());

policy = Box::new(RoundRobin::new());
```

同一个变量可以先拥有 `FirstHealthy`，再拥有 `RoundRobin`，因为变量的静态类型始终没有改变：

```text
变量类型始终是 Box<dyn Policy>
具体实现从 FirstHealthy 切换为 RoundRobin
```

重新赋值时，原来的 Box 及其内部策略会先被释放。

调用接收 `&mut dyn Policy` 的函数时，可以写：

```rust
select_with_dynamic_policy(policy.as_mut(), &workers)
```

类型变化为：

```text
Box<dyn Policy>
      │ .as_mut()
      ▼
&mut dyn Policy
```

`Box` 负责拥有策略，`&mut dyn Policy` 只在这次调用期间借用它。

### 何时适合 `Box<dyn Trait>`

- 根据配置在运行时选择一种实现；
- 容器需要保存多种实现，例如 `Vec<Box<dyn Middleware>>`；
- API 希望隐藏具体实现类型；
- 实现只有一个所有者，不需要跨多个组件共享所有权。

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
[`AtomicUsize` 与内部可变性](atomic-usize-and-interior-mutability.md)。

`Arc`、强弱引用计数与循环引用详见
[`Arc`、`Weak` 与循环引用](arc-weak-and-cycle-references.md)。

## `Box`、`Rc`、`Arc`、`Weak` 如何选择

| 类型 | 所有权 | 线程安全引用计数 | 典型用途 |
|---|---|---:|---|
| `&dyn Trait` | 不拥有，只共享借用 | 不适用 | 临时只读调用 |
| `&mut dyn Trait` | 不拥有，独占可变借用 | 不适用 | 临时调用需要修改状态的实现 |
| `Box<dyn Trait>` | 单一所有者 | 不使用引用计数 | 运行时多态、配置选择、异构容器 |
| `Rc<dyn Trait>` | 单线程共享所有权 | 否 | 单线程图结构或共享对象 |
| `Arc<dyn Trait>` | 多线程共享所有权 | 是 | 多任务、跨线程共享服务或策略 |
| `Weak<dyn Trait>` | 不拥有，可失效引用 | 与对应计数类型相关 | 回指、缓存、避免强引用环 |

选择时先问所有权问题，再问分发问题：

```text
只在调用期间使用？        -> &dyn Trait / &mut dyn Trait
需要一个长期独占所有者？  -> Box<dyn Trait>
单线程多个长期所有者？    -> Rc<dyn Trait>
跨线程多个长期所有者？    -> Arc<dyn Trait>
不应延长对象生命周期？    -> Weak<dyn Trait>
```

不是所有 trait 用法都需要智能指针。`P: Trait` 的静态分发通常直接保存具体值，不需要 Box 或
Arc。

## 动态分发与工厂方法

下面只是调用已有策略，不是工厂：

```rust
fn select_with_dynamic_policy(
    policy: &mut dyn Policy,
    workers: &[Worker],
) -> Option<usize>
```

真正的策略工厂会创建并返回策略：

```rust
fn create_policy(name: &str) -> Box<dyn Policy> {
    match name {
        "first-healthy" => Box::new(FirstHealthy::new()),
        "round-robin" => Box::new(RoundRobin::new()),
        _ => panic!("unknown policy"),
    }
}
```

教学项目后续应把未知配置改成明确的 `Result`，这里的重点只是观察返回类型：不同分支的具体
类型不同，但都能被擦除为 `Box<dyn Policy>`。

## Trait object 必须满足 dyn compatibility

不是所有 trait 都能直接写成 `dyn Trait`。编译器现在通常称这组规则为 dyn compatibility，
旧资料常称 object safety。

常见限制包括：

- trait 不能整体要求 `Self: Sized`；
- 要通过 trait object 调用的方法不能有自己的泛型类型参数；
- 方法不能在没有适当限制时按值返回未知的 `Self`；
- 方法需要具有可用于动态调用的接收者，例如 `&self`、`&mut self` 或 `Box<Self>`。

当前接口：

```rust
trait Policy {
    fn select(&mut self, workers: &[Worker]) -> Option<usize>;
}
```

可以构造 `dyn Policy`，因为它的方法满足动态调用所需条件。

如果编译器报告：

```text
the trait `...` is not dyn compatible
```

应先检查方法是否包含泛型参数、是否返回 `Self`，以及 trait 是否要求 `Sized`，而不是先尝试
加 Box 或 Arc。

## `Send` 和 `Sync` 是另一组约束

`dyn Policy` 只说明行为契约，不自动表示它能跨线程使用。需要跨线程时，API 往往明确要求：

```rust
Box<dyn Policy + Send>
Arc<dyn Policy + Send + Sync>
```

粗略理解：

- `Send`：值可以安全地转移到另一个线程；
- `Sync`：多个线程可以安全地共享对该值的引用。

具体实现也必须满足这些约束。`Arc` 的引用计数本身是线程安全的，但 `Arc<T>` 是否能在线程间
共享仍取决于内部 `T` 的能力。

## `my-smg`：配置驱动的共享策略

阶段 6 的网关根据配置选择 `FirstHealthy` 或 `RoundRobin`。两种具体类型不同，但需要放进同一个
字段，因此使用同一类型的 trait object：

```rust
let policy: Box<dyn Policy + Send> = match &config.policy {
    PolicyKind::FirstHealthy => Box::new(FirstHealthy::new()),
    PolicyKind::RoundRobin => Box::new(RoundRobin::new()),
};
```

| 组成 | 在这里的职责 |
|---|---|
| `Box` | 独占拥有选出的具体策略，给字段一个已知大小的指针类型 |
| `dyn Policy` | 擦除具体类型，通过 `Policy::select` 动态调用 |
| `+ Send` | 保证内部策略可安全地转移到其他线程 |

`Send` 不表示多个线程可以同时修改策略。`Policy::select(&mut self, ...)` 需要独占可变访问，
所以网关把它放在 `Mutex` 中，再把整个应用状态放进 `Arc`：

```text
多个请求 ──Arc──> GatewayState
                       └─ Mutex<Box<dyn Policy + Send>>
                                       └─ RoundRobin { next }
```

`Mutex<T>` 能在线程间共享的关键条件之一是 `T: Send`；这里的 `T` 正是
`Box<dyn Policy + Send>`。锁让每次 `select` 串行修改 `next`，`Arc` 让所有请求访问同一份
`GatewayState`。锁的取得、Guard 的释放见
[`Mutex`、`MutexGuard`、解引用与毒锁](mutex-guard-deref-and-poisoning.md)。

策略必须在请求之间保留状态。如果在每次 `chat` 请求中重新执行 `RoundRobin::new()`，`next`
每次都从 0 开始，有两个健康节点时会一直选第一个：

```text
共享同一个 RoundRobin：worker-1 → worker-2 → worker-1
每次请求重新创建：       worker-1 → worker-1 → worker-1
```

这里保存的是**轮询进度**。它不是“会话亲和性”：后者通常指同一客户端的请求持续进入同一节点。
`match &config.policy` 为什么借用字段，见[字段借用与 `match` 表达式](if-let-ref-and-deref.md#字段借用与-match-表达式)。

## 静态分发与动态分发对照

| 维度 | `P: Policy` / `impl Policy` | `dyn Policy` |
|---|---|---|
| 确定实现 | 编译期 | 运行时 |
| 调用方式 | 直接调用具体实现 | 通过虚表间接调用 |
| 内联机会 | 通常更好 | 通常较少 |
| 代码体积 | 每个具体类型可能单态化 | 调用路径更集中 |
| 同一变量运行时换实现 | 不可以直接换 | 配合 Box 等可以 |
| 异构容器 | 不直接支持 | `Vec<Box<dyn Policy>>` |
| 是否必须堆分配 | 否 | `&dyn Trait` 不需要；`Box<dyn Trait>` 需要 |
| 常见使用场景 | 算法、内部组件、性能敏感路径 | 插件、配置驱动实现、稳定抽象边界 |

不要仅凭“动态分发有一次间接调用”就提前优化。网关中的网络、序列化和排队成本通常远高于
一次虚表调用；应根据所有权、扩展性和实测结果选择。

## 常见编译错误

### `the size for values of type dyn Trait cannot be known`

原因通常是尝试按值保存裸 `dyn Trait`：

```rust
// 错误方向：裸 dyn Trait 没有编译期固定大小
// let policy: dyn Policy = ...;
```

改为根据所有权需求选择 `&dyn Policy`、`Box<dyn Policy>` 或 `Arc<dyn Policy>`。

### 找不到 trait 提供的方法

如果类型实现了 trait，但调用处报方法不存在，检查 trait 是否已进入当前作用域：

```rust
use crate::policy::Policy;
```

### 不能把 `&` 传给要求 `&mut` 的参数

trait 契约要求 `&mut self` 时，变量绑定和传参都必须允许可变借用：

```rust
let mut policy = RoundRobin::new();
select_with_policy(&mut policy, &workers);
```

### `Arc` 中的值不能可变借用

`Arc<T>` 只允许多个所有者共享访问。若行为需要修改状态，应选择原子类型、锁或新的状态边界，
不能把 `Arc` 当成自动提供 `&mut T` 的工具。

## 当前学习顺序

```text
两个具体策略
    -> 抽取 Policy trait
    -> P: Policy 静态分发
    -> &mut dyn Policy 动态借用
    -> Box<dyn Policy> 拥有并切换实现
    -> 并发阶段再学习 Arc、AtomicUsize、Send、Sync
```

这个顺序的重点是先遇到实际需求，再引入对应抽象。当前单线程轮询器不需要为了模仿上游而
提前使用 `Arc` 或原子变量。

## 一句话记忆

```text
trait：大家承诺会做什么。
泛型：编译器知道具体是谁，分别生成调用代码。
dyn：调用时才知道具体是谁，通过虚表找到实现。
Box：我独占拥有这个实现。
Arc：我们共同拥有这个实现。
Weak：我能找到它，但不负责让它存活。
```
