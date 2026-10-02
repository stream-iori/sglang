# Rust 学习笔记索引

本目录记录阅读和实现 SGLang / SGL Model Gateway 时遇到的 Rust 概念。文章按主题独立编写，
建议按下面的路线逐步阅读，而不是一次记住全部语法。

## 推荐学习路线

```text
Cargo 与 crate
      │
      v
模块与可见性
      │
      v
所有权、move、借用
      │
      ├─> 切片与 Vec 类型推断
      ├─> 模式解构、if let、let ... else、ref、解引用
      ├─> Result、?、错误传播
      ├─> Formatter、生命周期、Error
      └─> derive、Copy、Clone 等 trait
                    │
                    v
          trait 与静态/动态分发
                    │
                    v
             Box / Arc / Weak
                    │
                    v
          Mutex / MutexGuard / 解引用
                    │
                    v
          AtomicUsize / 内部可变性
                    │
                    v
          Tokio / 异步任务 / 取消
```

## 基础结构

### [学习感悟：从运行时需求反推所有权设计](realization.md)

以动态删除节点为例，记录“可选资格与对象存活分离”的理解；关联 Arc 的技术细节与软件设计原则。

### [Rust 软件设计的第一性原理](desgin-principle.md)

用 `my-smg` 的配置加载和错误体系串联：失败建模、分层边界、信息保留、所有权、关注点分离、
结构化错误、fail-fast，以及“按真实需求引入复杂度”。适合学完具体语法后回看设计理由。

### [Cargo Workspace、Package 与 Crate](workspace.md)

区分 workspace、package 和 crate，理解 `Cargo.toml`、workspace member、统一依赖、profile 以及
常用 Cargo 命令；同时解释 `src/main.rs`、`src/bin/*.rs` 和 `[[bin]]` 如何定义二进制 target。
适合作为本目录第一篇。

### [`crate`、`mod`、`pub`、`use` 与模块路径](modules-visibility-and-crate.md)

解释 Rust 如何组织代码、`src/lib.rs` 与 `src/main.rs` 的角色、模块树、可见性以及重新导出。
开始拆分源码文件时阅读。

## 所有权与数据访问

### [所有权、转移与借用](ownership-move-and-borrowing.md)

系统介绍 move、`&T`、`&mut T`、方法接收者、`Copy`、`Clone`、部分移动、容器迭代和 `Arc` 的
所有权含义；并用 `InFlightGuard<'a>` 解释借用检查如何防止 guard 比 Worker 活得更久。
这是理解后续文章的核心基础。

### [`&[Worker]`、`Vec::<usize>::new()` 与 turbofish](borrowed-slices-and-vec-type-annotation.md)

结合节点选择场景解释借用切片、数组与 `Vec` 的区别、turbofish 类型标注、空 `Vec` 的类型
推断，以及为什么索引使用 `usize`；对照 parse、Vec、collect、pending 的泛型参数位置，
并拆解 `pending::<()>()` 中两个 `()` 的不同职责。

### [模式解构、`if let`、`let ... else`、`ref` 与解引用](if-let-ref-and-deref.md)

解释 `let Some(index) = value else { ... };` 如何提前返回、`State(state)` 如何解构元组
结构体、`&config.policy` 如何借用字段、`match` 分号如何影响返回值，以及锁守卫的解引用。

### [`Result`、`?` 与错误传播](result-question-mark-and-error-propagation.md)

解释 `Result<T, E>`、`?` 的成功与失败控制流、“把错误交给调用者”的准确含义、错误类型转换，
以及 `?`、`match`、`unwrap`、`expect` 的使用边界。结合 JSON 配置解析理解语法错误与业务校验
错误的区别。

## Trait 与类型能力

### [常见 `derive` 速查](common-derive.md)

覆盖 `Debug`、`Clone`、`Copy`、`Default`、比较、排序、哈希、Serde、thiserror 和 clap，并解释
泛型类型的 trait bound。需要为结构体或枚举添加能力时按章节查阅。

### [`Formatter<'_>`、生命周期、`Display` 与 `Error`](formatter-lifetimes-display-and-error.md)

从 `ConfigError` 的标准库实现出发，解释 `Formatter<'_>` 的两层借用、生命周期参数与 `'_`
推断、`Display` 格式化管线、`Error: Debug + Display`、`source()`、`dyn Error`，以及 `Error`、
`From`、`?` 三者的分工。

### [Trait、静态分发、动态分发与智能指针](traits-dispatch-and-smart-pointers.md)

从 `Policy` 策略接口出发，串联：

- trait 与 Java interface 的异同；
- `P: Trait`、`impl Trait` 和单态化；
- `dyn Trait`、虚表与 dyn compatibility；
- `&dyn Trait`、`Box<dyn Trait>`、`Rc<dyn Trait>`、`Arc<dyn Trait>` 的所有权差异；
- `Send`、`Sync` 和共享可变状态的边界；
- 静态分发、动态分发和工厂方法的区别。
- `my-smg` 中 `Box<dyn Policy + Send>` 如何与 `Arc`、`Mutex` 一起保留轮询进度。

建议先读所有权文章，再读本篇。

### [闭包、Fn 系列与条件等待](closures-and-fn-traits.md)

从 `|| worker.counter() == 1` 开始，解释无参数闭包、捕获变量、Fn / FnMut / FnOnce、
参数中的 `impl FnMut() -> bool`，以及路由处理函数里的外层 move 与内层 async move。
包含三种闭包的小例子和状态变化对照。

## 共享所有权与生命周期

### [`Arc`、`Weak` 与循环引用](arc-weak-and-cycle-references.md)

深入解释强引用与弱引用、循环引用为什么导致对象无法释放、`Weak::upgrade()`，以及 SGL Model
Gateway 中的实际父子关系；也解释节点从注册表移除后，为何请求的 Arc 仍能保住对象。
读完 trait object 与 `Arc<dyn Trait>` 后再读更容易理解。

### [`Mutex`、`MutexGuard`、解引用与毒锁](mutex-guard-deref-and-poisoning.md)

结合 `my-smg` 的失败计数器解释 `Arc<Mutex<T>>`、`lock()` 的 `Result`、毒锁恢复、
`MutexGuard<'_, T>`、`Deref/DerefMut`、RAII 自动解锁，以及为什么不能持有 Guard 跨越 `.await`。

### [`AtomicUsize` 与内部可变性](atomic-usize-and-interior-mutability.md)

结合每节点在途计数解释为何原子变量能通过 `&self` 更新、`fetch_add` 返回什么、
`Ordering::Relaxed` 保留了什么保证，以及与 `Arc`、`Mutex` 的分工和减一的边界。

## 异步与 Tokio

### [Tokio 专题：异步基础、任务句柄与取消](tokio-basics-and-task-cancellation.md)

先区分 Rust 的 Future、async/await 与 Tokio 的 Runtime、任务、线程，再解释 spawn、
async move、JoinHandle、sleep、timeout、pending 和取消测试；包含 Java 类比及易混淆的退出规则。
第 4 节集中解释捕获推断、何时需要 move、Future 暂停时保存变量、Send / static 约束及释放过程。
第 7.1 节用测试重构的 ManagedTask / TestServer 关联泛型、消费自身、take、Drop 和显式异步清理。

## 按问题查找

| 遇到的问题 | 建议文章 |
|---|---|
| workspace、package、crate 分不清 | [Cargo Workspace、Package 与 Crate](workspace.md) |
| Cargo 如何识别 `src/bin/fake-worker.rs` | [Cargo 如何识别 library 和 binary](workspace.md#cargo-如何识别-library-和-binary) |
| 为什么要这样设计错误和配置边界 | [Rust 软件设计的第一性原理](desgin-principle.md) |
| `pub mod`、`crate::`、`use` 看不懂 | [模块与可见性](modules-visibility-and-crate.md) |
| `use of moved value` | [所有权、转移与借用](ownership-move-and-borrowing.md) |
| `&T` 与 `&mut T` 不清楚 | [所有权、转移与借用](ownership-move-and-borrowing.md) |
| `mut self` 是不是 `&mut self`，take 是不是 clone | [消费自身与 Option::take](ownership-move-and-borrowing.md#consuming-self-and-option-take) |
| `InFlightGuard<'a>` 为什么不能比 Worker 活得更久 | [guard 借用 Worker 与生命周期检查](ownership-move-and-borrowing.md#guard-借用-worker生命周期如何防止先销毁被借用者) |
| `&[Worker]` 是什么 | [借用切片与 Vec 类型标注](borrowed-slices-and-vec-type-annotation.md) |
| 空 `Vec` 无法推断类型 | [借用切片与 Vec 类型标注](borrowed-slices-and-vec-type-annotation.md) |
| turbofish 干什么，常见场景有哪些 | [turbofish：给泛型指定类型](borrowed-slices-and-vec-type-annotation.md#turbofish给泛型指定类型) |
| `parse::<u64>()` 看不懂 | [`parse::<u64>()`：给泛型方法指定目标类型](borrowed-slices-and-vec-type-annotation.md#parseu64给泛型方法指定目标类型) |
| `pending::<()>()` 中两个 `()` 什么意思 | [pending：两个括号不同](borrowed-slices-and-vec-type-annotation.md#pending泛型类型与函数调用) |
| `State(state): State<AppState>` 看不懂 | [`State(state)`：解构元组结构体](if-let-ref-and-deref.md#statestate解构元组结构体) |
| `let Some(index) = value else` 看不懂 | [`let ... else`：取出值，失败就提前离开](if-let-ref-and-deref.md#let--else取出值失败就提前离开) |
| `&config.policy` 借用哪个值 | [字段借用与 `match` 表达式](if-let-ref-and-deref.md#字段借用与-match-表达式) |
| `match` 后的分号为什么改变返回值 | [字段借用与 `match` 表达式](if-let-ref-and-deref.md#字段借用与-match-表达式) |
| `Err(error) if error.is_cancelled()` 怎么匹配 | [match 分支守卫](if-let-ref-and-deref.md#match-guards) |
| `Some(ref value)` 或锁守卫解引用 | [`if let`、`ref` 与解引用](if-let-ref-and-deref.md) |
| `?` 为什么会提前返回 | [`Result`、`?` 与错误传播](result-question-mark-and-error-propagation.md) |
| `.into()` 如何推断目标，From 是不是反向转换 | [Into、From 与可失败转换](result-question-mark-and-error-propagation.md#类型转换intofromtryinto-与-tryfrom) |
| `try_into()` 与 `into()` 有什么区别 | [可能失败的转换](result-question-mark-and-error-propagation.md#可能失败的转换) |
| `Result`、`unwrap`、`expect` 如何选择 | [`Result`、`?` 与错误传播](result-question-mark-and-error-propagation.md) |
| `Formatter<'_>` 和 `'_` 是什么 | [Formatter、生命周期、Display 与 Error](formatter-lifetimes-display-and-error.md) |
| `Display` 与 `Debug` 有什么区别 | [Formatter、生命周期、Display 与 Error](formatter-lifetimes-display-and-error.md) |
| 实现 `Error` 到底有什么用 | [Formatter、生命周期、Display 与 Error](formatter-lifetimes-display-and-error.md) |
| `Error`、`From`、`?` 如何配合 | [Formatter、生命周期、Display 与 Error](formatter-lifetimes-display-and-error.md) |
| `From` 与 `source()` 有什么区别 | [From 与 source 对照](formatter-lifetimes-display-and-error.md#from-与-source-不要混淆) |
| 不知道该派生哪些 trait | [常见 derive 速查](common-derive.md) |
| `P: Trait` 与 `dyn Trait` 的区别 | [Trait 与分发](traits-dispatch-and-smart-pointers.md) |
| `ManagedTask<T>` 和 `impl<T>` 中的 T 是什么 | [泛型结构体与实现](traits-dispatch-and-smart-pointers.md#generic-struct-and-impl) |
| `||`、`impl FnMut() -> bool` 怎么读 | [闭包与 Fn 系列](closures-and-fn-traits.md#fn-family) |
| 外层 move 和内层 async move 为什么能重复调用 | [双层捕获与 Copy](closures-and-fn-traits.md#double-move) |
| 为什么需要 `Box<dyn Trait>` | [Trait 与分发](traits-dispatch-and-smart-pointers.md) |
| `Box<dyn Policy + Send>` 为什么放进 `Mutex` | [配置驱动的共享策略](traits-dispatch-and-smart-pointers.md#my-smg配置驱动的共享策略) |
| 每次 `RoundRobin::new()` 为什么只选第一个 | [配置驱动的共享策略](traits-dispatch-and-smart-pointers.md#my-smg配置驱动的共享策略) |
| `Arc` 为什么不能直接修改内部值 | [Trait 与分发](traits-dispatch-and-smart-pointers.md) |
| `Arc` 循环引用或 `Weak::upgrade()` | [`Arc`、`Weak` 与循环引用](arc-weak-and-cycle-references.md) |
| `Arc::ptr_eq` 与值相等有什么区别 | [对象身份与值相等](arc-weak-and-cycle-references.md#ptr_eq对象身份与值相等) |
| 节点已经移除，Arc 保住的到底是什么 | [节点移除与对象销毁](arc-weak-and-cycle-references.md#节点移除与对象销毁是两件事) |
| 从运行时视角理解关注点分离 | [学习感悟 realization](realization.md) |
| `MutexGuard<'_, T>` 是什么 | [`Mutex`、`MutexGuard`、解引用与毒锁](mutex-guard-deref-and-poisoning.md) |
| 为什么要写 `*guard` | [MutexGuard 为什么能解引用](mutex-guard-deref-and-poisoning.md#mutexguard-为什么能用--解引用) |
| `lock()` 为什么返回 `Result` | [`lock()` 与毒锁](mutex-guard-deref-and-poisoning.md#lock-为什么返回-result) |
| 为什么不能拿着锁 `.await` | [Guard 与 `.await`](mutex-guard-deref-and-poisoning.md#为什么不能持有-guard-跨越-await) |
| `AtomicUsize` 为什么能通过 `&self` 加一 | [`AtomicUsize` 与内部可变性](atomic-usize-and-interior-mutability.md) |
| `Ordering::Relaxed` 是不是不再原子 | [`Relaxed` 到底放松了什么](atomic-usize-and-interior-mutability.md#relaxed-到底放松了什么) |
| Runtime、Future、async/await、任务与线程分不清 | [Tokio 基础专题](tokio-basics-and-task-cancellation.md) |
| `async move` 能否省略，什么时候需要 move | [捕获推断与 move](tokio-basics-and-task-cancellation.md#42-不写-move也可能移动变量) |
| Future 暂停后，变量和 guard 为什么还在 | [Future 保存执行状态](tokio-basics-and-task-cancellation.md#44-future-持有变量为什么暂停后仍存在) |
| move 为什么不保证 Send，static 是否表示永远存活 | [Send 与 static 的边界](tokio-basics-and-task-cancellation.md#45-arcsend-与-static-的边界) |
| JoinHandle 来自标准库还是 Tokio，像不像 Runnable | [JoinHandle：不是 Runnable](tokio-basics-and-task-cancellation.md#5-joinhandle不是-runnable) |
| timeout 中检查计数的循环是什么意思 | [sleep、timeout 和 pending](tokio-basics-and-task-cancellation.md#6-sleeptimeout-和-pending) |
| abort 后为什么还要 await，guard 为什么会 Drop | [取消测试完整过程](tokio-basics-and-task-cancellation.md#7-abort-与-drop取消测试完整过程) |
| TestServer 没有实现 Drop，为什么仍会请求取消 | [组合对象与任务资源管理](tokio-basics-and-task-cancellation.md#managed-task-resource-lifecycle) |
| 测试该抽哪些辅助逻辑，断言放在哪里 | [测试重构的设计原则](desgin-principle.md#test-refactor-principles) |

## 阅读原则

- 先根据编译错误或当前代码寻找具体问题，再阅读对应章节。
- 区分“所有权选择”和“分发选择”：`Box/Arc` 回答谁拥有，泛型/`dyn` 回答如何调用。
- 不因为上游代码使用了 `Arc`、原子变量或 trait object，就提前复制复杂度。
- 每学一个概念，优先写一个可观察行为的最小测试，而不是只记语法。
