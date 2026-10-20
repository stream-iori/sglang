# 生命周期：引用不能比数据活得久

[分类目录](index.md) · [总入口](../../index.md) · [问题索引](../../questions.md)

**生命周期标注表达借用关系，不会让数据多活一秒。**

| 阅读准备 | 内容 |
|---|---|
| 前置 | [共享借用与可变借用](borrowing.md) |
| 源码入口 | [src/worker.rs](../../../../src/worker.rs)：`InFlightGuard<'a>` |
| 阅读时机 | T06 按需读到基础自查；后续各节按标注阶段阅读。 |

## 借用会在最后一次使用后结束

借用的有效范围不一定持续到整个 `{}` 结束：

```rust
let mut text = String::from("hello");

let reader = &text;
println!("{reader}"); // reader 最后一次使用

let writer = &mut text; // 可以
writer.push('!');
```

编译器发现 `reader` 后面不再使用，就允许接着创建可变借用。这称为 non-lexical lifetimes
（NLL）。

如果可变借用后还要直接使用原变量，确保借用已经不再使用：

```rust
let mut values = vec![1, 2];

let first = &mut values[0];
*first += 10;
// first 后面不再使用，借用结束

values.push(3);
```

## 返回引用先找数据来源

Worker::id 的 &self → &str 表示借出节点已经拥有的文本。
若函数创建局部 String 再返回它的引用，函数结束后就没有拥有者托住文本；
生命周期标注不能延长资源存活。

```text
调用者拥有 → 函数借用 → 返回引用仍受原数据约束
局部值拥有 → 函数结束销毁 → 不能交出悬空引用
```

## 基础自查

| 类型 | 问题 |
|---|---|
| 预测 | reader 最后一次使用后可以创建 writer 吗？ |
| 解释 | 为什么生命周期标注不能延长局部值的存活？ |
| 边界 | 把 reader 的一次使用移到 writer 修改之后，仍能通过吗？ |

先写下判断与原因，再用自己已有的代码验证；遇到疑问回查本篇，不需要提前读后续应用。

## 阶段 4：Formatter 的生命周期参数

先认识[trait](../types/traits-and-bounds.md)和[Display 职责](../errors/debug-display-and-formatter.md)。签名省略了实现体。

显式命名生命周期时，可以写：

```rust
fn render<'a>(formatter: &mut fmt::Formatter<'a>) -> fmt::Result {
    // ...
}
```

拆开看：

```text
<'a>                    声明一个生命周期参数
Formatter<'a>           Formatter 内部缓冲区借用满足 'a
&mut Formatter<'a>      当前函数临时可变借用这个 Formatter
```

`'a` 不是具体的秒数，也不表示“整个函数生命周期”。它是编译器用来比较引用有效范围的名字。
想看借用如何在实际代码中阻止“先销毁被借用者”，可读
[guard 借用 Worker：生命周期如何防止先销毁被借用者](lifetimes.md)。

## 阶段 4：`'_`：让编译器推断生命周期

在 `Display` 实现里通常写：

```rust
fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result
```

`'_` 表示：

```text
这里必须有一个生命周期
但当前代码不需要给它命名
请编译器根据调用关系推断
```

它不表示：

- 忽略生命周期；
- 生命周期无限长；
- `'static`；
- 创建一个新对象。

显式命名与占位写法表达的核心约束相同：

```rust
fn fmt<'a>(&self, f: &mut fmt::Formatter<'a>) -> fmt::Result
```

```rust
fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result
```

这里只需要把 Formatter 用于当前调用，没有把其中的引用返回或保存，所以 `'_` 更简洁。

## 阶段 4：生命周期省略与 `'_` 的区别

```rust
fn id(&self) -> &str
```

这里没有写生命周期，因为生命周期省略规则能判断返回引用来自 `self`。概念上接近：

```rust
fn id<'a>(&'a self) -> &'a str
```

而 `Formatter<'_>` 中，`Formatter` 这个类型本身声明了一个生命周期参数。写出 `'_` 是明确
告诉读者和编译器：“这个泛型参数是生命周期，请在这里推断”。

```text
省略生命周期 -> 语法规则自动补全
'_           -> 显式放置一个匿名生命周期占位符
```

两者都依赖编译器推断，但出现位置和表达目的不同。

## 阶段 4：`'static` 不等于“这个引用永远存在”

`Error::source` 的常见签名是：

```rust
fn source(&self) -> Option<&(dyn Error + 'static)>
```

这里有两个不同概念：

```text
返回的 & 引用
└─ 生命周期省略后与 &self 绑定，不会超过 self

dyn Error + 'static
└─ trait object 内的具体错误不能借用短生命周期数据
```

因此 `+ 'static` 不是说返回的引用可以永远使用。它约束的是被引用错误对象内部不能依赖短期
借用。返回引用仍然受当前错误对象的生命周期限制。

拥有 `String`、`Vec` 等数据的错误通常可以满足 `'static`；保存 `&str` 的错误则可能需要额外
生命周期参数，且未必满足这一约束。

## 项目应用：阶段 7 的 guard 与借用

另需[Drop](drop-and-raii.md)基础部分。

`my-smg` 的在途计数守卫保存了一个引用，而不是拥有整个节点：

```rust
pub struct InFlightGuard<'a> {
    worker: &'a Worker,
}

impl Worker {
    pub fn begin_request(&self) -> InFlightGuard<'_> {
        self.increment_count();
        InFlightGuard { worker: self }
    }
}
```

把返回类型中的 `'_` 显式命名，方法的借用关系可以写成：

```rust
pub fn begin_request<'a>(&'a self) -> InFlightGuard<'a>
```

这不是延长 `Worker` 的寿命，而是向编译器说明：guard 中的引用来自这次对 `self` 的借用。
编译器必须确认 `Worker` 在 guard 可能使用该引用的整个期间都有效。
以下片段使用当前库接口；先引入 my_smg::worker::Worker，再放入函数体。
它不是独立完整程序，预期诊断为 E0505，而不是缺少参数或符号：

```rust,compile_fail
let worker = Worker::new("a".to_string(), "http://127.0.0.1:3001".to_string());
let guard = worker.begin_request();

drop(worker); // 错：worker 仍被 guard 借用，不能先移动并销毁
drop(guard);
```

换成先 `drop(guard)`、再 `drop(worker)` 才满足借用关系。`'a` 是编译期约束，
不是运行时计时器，也不会替你保留一个本来已经被销毁的对象。
若运行时要求“列表立即移除节点，但已有请求继续持有对象”，需要区别借用与独立所有权，见
[Arc：节点移除与对象销毁](../../walkthroughs/worker-lifecycle.md)。

这里与上一节的 NLL 并不矛盾：`InFlightGuard` 实现了 `Drop`，销毁时还要通过 `&Worker`
把计数减一。即使源码中没有再次显式读取 `guard.worker`，编译器仍要保证该引用在 guard
销毁时有效。

`InFlightGuard` 与标准库的 `MutexGuard` 没有类型上的继承关系；两者只是都利用 `Drop`
自动完成收尾：前者减计数，后者解锁。计数器的原子操作见
[`AtomicUsize` 与内部可变性](../sharing/atomics.md)，锁守卫见
[`Mutex`、`MutexGuard`、解引用与毒锁](../sharing/mutex-and-guards.md)。

## 阶段 7：&'static T 和 T: 'static 有什么区别

| 写法 | 约束什么 | 不代表什么 |
|---|---|---|
| `&'static str` | 引用指向的字符串在整个程序期间有效 | 保存引用的局部变量必须一直存在 |
| `T: 'static` | 类型 T 不依赖短生命周期的借用 | T 的值永远不能被销毁 |

拥有数据的 `String` 通常满足 `T: 'static`，但仍会在离开作用域时正常释放。
局部 String 的 `&String` 则不能仅靠加类型标注变成 `'static` 引用。

## `missing lifetime specifier`

编译器无法判断返回引用来自哪个输入。应先画出“谁拥有数据、谁借用数据”，再补生命周期，
不要机械添加 `'static`。

## 自查与关联

综合题（阶段 7）：guard 中的 &'a Worker 限制了什么？为什么先释放 guard、再释放 worker 的正例能通过？

[Move 与部分移动](move-and-partial-move.md) · [共享借用与可变借用](borrowing.md) · [self、mut self、&self 与 &mut self](method-receivers.md)
