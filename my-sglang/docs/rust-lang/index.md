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
```

## 基础结构

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
所有权含义。这是理解后续文章的核心基础。

### [`&[Worker]`、`Vec::<usize>::new()` 与 turbofish](borrowed-slices-and-vec-type-annotation.md)

结合节点选择场景解释借用切片、数组与 `Vec` 的区别、turbofish 类型标注、空 `Vec` 的类型
推断，以及为什么索引使用 `usize`；同时对比 `Vec::<usize>::new()` 和 `parse::<u64>()`。

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

## 共享所有权与生命周期

### [`Arc`、`Weak` 与循环引用](arc-weak-and-cycle-references.md)

深入解释强引用与弱引用、循环引用为什么导致对象无法释放、`Weak::upgrade()`，以及 SGL Model
Gateway 中的实际父子关系。读完 trait object 与 `Arc<dyn Trait>` 后再读更容易理解。

### [`Mutex`、`MutexGuard`、解引用与毒锁](mutex-guard-deref-and-poisoning.md)

结合 `my-smg` 的失败计数器解释 `Arc<Mutex<T>>`、`lock()` 的 `Result`、毒锁恢复、
`MutexGuard<'_, T>`、`Deref/DerefMut`、RAII 自动解锁，以及为什么不能持有 Guard 跨越 `.await`。

## 按问题查找

| 遇到的问题 | 建议文章 |
|---|---|
| workspace、package、crate 分不清 | [Cargo Workspace、Package 与 Crate](workspace.md) |
| Cargo 如何识别 `src/bin/fake-worker.rs` | [Cargo 如何识别 library 和 binary](workspace.md#cargo-如何识别-library-和-binary) |
| 为什么要这样设计错误和配置边界 | [Rust 软件设计的第一性原理](desgin-principle.md) |
| `pub mod`、`crate::`、`use` 看不懂 | [模块与可见性](modules-visibility-and-crate.md) |
| `use of moved value` | [所有权、转移与借用](ownership-move-and-borrowing.md) |
| `&T` 与 `&mut T` 不清楚 | [所有权、转移与借用](ownership-move-and-borrowing.md) |
| `&[Worker]` 是什么 | [借用切片与 Vec 类型标注](borrowed-slices-and-vec-type-annotation.md) |
| 空 `Vec` 无法推断类型 | [借用切片与 Vec 类型标注](borrowed-slices-and-vec-type-annotation.md) |
| `parse::<u64>()` 或 turbofish 看不懂 | [`parse::<u64>()`：给泛型方法指定目标类型](borrowed-slices-and-vec-type-annotation.md#parseu64给泛型方法指定目标类型) |
| `State(state): State<AppState>` 看不懂 | [`State(state)`：解构元组结构体](if-let-ref-and-deref.md#statestate解构元组结构体) |
| `let Some(index) = value else` 看不懂 | [`let ... else`：取出值，失败就提前离开](if-let-ref-and-deref.md#let--else取出值失败就提前离开) |
| `&config.policy` 借用哪个值 | [字段借用与 `match` 表达式](if-let-ref-and-deref.md#字段借用与-match-表达式) |
| `match` 后的分号为什么改变返回值 | [字段借用与 `match` 表达式](if-let-ref-and-deref.md#字段借用与-match-表达式) |
| `Some(ref value)` 或锁守卫解引用 | [`if let`、`ref` 与解引用](if-let-ref-and-deref.md) |
| `?` 为什么会提前返回 | [`Result`、`?` 与错误传播](result-question-mark-and-error-propagation.md) |
| `Result`、`unwrap`、`expect` 如何选择 | [`Result`、`?` 与错误传播](result-question-mark-and-error-propagation.md) |
| `Formatter<'_>` 和 `'_` 是什么 | [Formatter、生命周期、Display 与 Error](formatter-lifetimes-display-and-error.md) |
| `Display` 与 `Debug` 有什么区别 | [Formatter、生命周期、Display 与 Error](formatter-lifetimes-display-and-error.md) |
| 实现 `Error` 到底有什么用 | [Formatter、生命周期、Display 与 Error](formatter-lifetimes-display-and-error.md) |
| `Error`、`From`、`?` 如何配合 | [Formatter、生命周期、Display 与 Error](formatter-lifetimes-display-and-error.md) |
| `From` 与 `source()` 有什么区别 | [From 与 source 对照](formatter-lifetimes-display-and-error.md#from-与-source-不要混淆) |
| 不知道该派生哪些 trait | [常见 derive 速查](common-derive.md) |
| `P: Trait` 与 `dyn Trait` 的区别 | [Trait 与分发](traits-dispatch-and-smart-pointers.md) |
| 为什么需要 `Box<dyn Trait>` | [Trait 与分发](traits-dispatch-and-smart-pointers.md) |
| `Box<dyn Policy + Send>` 为什么放进 `Mutex` | [配置驱动的共享策略](traits-dispatch-and-smart-pointers.md#my-smg配置驱动的共享策略) |
| 每次 `RoundRobin::new()` 为什么只选第一个 | [配置驱动的共享策略](traits-dispatch-and-smart-pointers.md#my-smg配置驱动的共享策略) |
| `Arc` 为什么不能直接修改内部值 | [Trait 与分发](traits-dispatch-and-smart-pointers.md) |
| `Arc` 循环引用或 `Weak::upgrade()` | [`Arc`、`Weak` 与循环引用](arc-weak-and-cycle-references.md) |
| `MutexGuard<'_, T>` 是什么 | [`Mutex`、`MutexGuard`、解引用与毒锁](mutex-guard-deref-and-poisoning.md) |
| 为什么要写 `*guard` | [MutexGuard 为什么能解引用](mutex-guard-deref-and-poisoning.md#mutexguard-为什么能用--解引用) |
| `lock()` 为什么返回 `Result` | [`lock()` 与毒锁](mutex-guard-deref-and-poisoning.md#lock-为什么返回-result) |
| 为什么不能拿着锁 `.await` | [Guard 与 `.await`](mutex-guard-deref-and-poisoning.md#为什么不能持有-guard-跨越-await) |

## 阅读原则

- 先根据编译错误或当前代码寻找具体问题，再阅读对应章节。
- 区分“所有权选择”和“分发选择”：`Box/Arc` 回答谁拥有，泛型/`dyn` 回答如何调用。
- 不因为上游代码使用了 `Arc`、原子变量或 trait object，就提前复制复杂度。
- 每学一个概念，优先写一个可观察行为的最小测试，而不是只记语法。
