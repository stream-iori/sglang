# Rust：所有权、转移与借用

## 一句话

Rust 用所有权管理资源，不需要垃圾回收器：

```text
一个值有一个所有者
所有者离开作用域时，值被释放
所有权可以 move，也可以暂时借给别人
```

最常用的选择是：

```text
把值交给下游，不再使用 -> move
只让下游读取           -> &T
让下游原地修改         -> &mut T
双方都必须拥有独立值   -> clone()
简单小值允许隐式复制   -> Copy
```

## 所有权解决什么问题

以 `String` 为例，它在栈上保存指针、长度和容量，真正的字符存放在堆上：

```rust
let text = String::from("hello");
```

```text
栈                         堆
text
├─ ptr  -----------------> "hello"
├─ len = 5
└─ capacity = 5
```

变量 `text` 拥有这个 `String`。当 `text` 离开作用域时，Rust 自动调用 `drop`，释放堆内存：

```rust
{
    let text = String::from("hello");
    println!("{text}");
} // text 离开作用域，String 在这里被释放
```

编译器需要始终知道谁负责释放资源，这就是所有权规则的目的。

## move：转移所有权，不复制数据

```rust
let first = String::from("hello");
let second = first;

println!("{second}");
// println!("{first}"); // 编译错误：first 已经被 move
```

`let second = first` 没有复制堆上的字符串，只把所有权交给 `second`：

```text
转移前：first  ──拥有──> String("hello")

转移后：first  ──失效
        second ──拥有──> 原来的 String("hello")
```

如果 `first` 仍然有效，两个变量就会在离开作用域时释放同一块内存。Rust 让旧变量失效，从编译期
避免 double free。

### move 不只发生在赋值时

按值传参也会转移所有权：

```rust
fn consume(text: String) {
    println!("{text}");
} // text 在函数结束时释放

let message = String::from("capture payment");
consume(message);

// println!("{message}"); // 编译错误：所有权已经传给 consume
```

放入容器同样会 move：

```rust
let message = String::from("capture payment");
let mut messages = Vec::new();

messages.push(message);
// println!("{message}"); // 编译错误：String 现在由 Vec 拥有
```

闭包或异步任务使用 `move` 时，也会取得捕获值的所有权：

```rust
let request_id = String::from("req-100");

let task = move || {
    println!("{request_id}");
};

task();
// request_id 已经属于闭包
```

`move` 关键字表示闭包按值捕获；如果捕获的类型实现了 `Copy`，实际得到的是副本。

异步块同样需要区分捕获方式：`async { ... }` 可以推断借用或移动，`async move { ... }`
明确按值捕获。move 引用不延长被引用对象的存活时间，move Arc 也不自动 clone。
相关例子和 spawn 的生命周期要求见
[Tokio 专题：不写 move，也可能移动变量](tokio-basics-and-task-cancellation.md#42-不写-move也可能移动变量)。

## 函数返回值也能转移所有权

```rust
fn create_command() -> String {
    let command = String::from("capture payment");
    command
}

let command = create_command();
```

关系是：

```text
create_command 中的 command
              │ move
              ▼
调用方的 command
```

返回 `String` 不会因为局部变量离开作用域而产生悬垂指针；所有权已经移动给调用方。现代 Rust
还会使用返回值优化，通常不需要为了性能改成复杂的输出参数。

如果函数消费一个值后还要把它交还，也可以返回它：

```rust
fn normalize(mut text: String) -> String {
    text.make_ascii_lowercase();
    text
}

let name = String::from("PAYMENT");
let name = normalize(name);
```

但函数只需要临时使用值时，借用通常更合适。

## 借用 `&T`：暂时读取，不取得所有权

```rust
fn print_length(text: &String) {
    println!("{}", text.len());
}

let message = String::from("hello");

print_length(&message);
print_length(&message);
println!("{message}");
```

`&message` 创建共享引用：

```text
message ──拥有──> String("hello")
                    ▲
                    │ 借用
text: &String ------┘
```

函数只借来看，结束后所有权仍属于 `message`。

字符串参数通常优先写成 `&str`，这样既能接收 `String`，也能接收字符串字面量：

```rust
fn validate_order_no(order_no: &str) -> bool {
    !order_no.is_empty()
}

let owned = String::from("ORDER-100");

assert!(validate_order_no(&owned));
assert!(validate_order_no("ORDER-200"));
```

## 可变借用 `&mut T`：暂时独占并修改

```rust
fn mark_captured(status: &mut String) {
    status.clear();
    status.push_str("CAPTURED");
}

let mut status = String::from("AUTHORIZED");
mark_captured(&mut status);

assert_eq!(status, "CAPTURED");
```

`&mut status` 没有取得 `String` 的所有权，但在借用期间拥有独占访问权。

### 多个共享借用，或者一个可变借用

同一时刻可以有多个 `&T`：

```rust
let text = String::from("hello");

let first = &text;
let second = &text;

println!("{first} {second}");
```

但不能让共享借用和可变借用同时被使用：

```rust
let mut text = String::from("hello");

let reader = &text;
// let writer = &mut text; // 如果后面还使用 reader，这里会编译失败

println!("{reader}");
```

也不能同时使用两个可变借用：

```rust
let mut text = String::from("hello");

let first = &mut text;
// let second = &mut text; // first 后面仍会使用，因此编译失败

first.push('!');
```

规则可以记成：

```text
同一时刻：
任意多个只读者 &T
或者
一个写入者 &mut T
```

它在编译期避免数据竞争和迭代过程中修改容器等问题。

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

## guard 借用 Worker：生命周期如何防止先销毁被借用者

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
编译器必须确认 `Worker` 在 guard 可能使用该引用的整个期间都有效：

```rust,compile_fail
let worker = Worker::new("a".to_string());
let guard = worker.begin_request();

drop(worker); // 错：worker 仍被 guard 借用，不能先移动并销毁
drop(guard);
```

换成先 `drop(guard)`、再 `drop(worker)` 才满足借用关系。`'a` 是编译期约束，
不是运行时计时器，也不会替你保留一个本来已经被销毁的对象。
若运行时要求“列表立即移除节点，但已有请求继续持有对象”，需要区别借用与独立所有权，见
[Arc：节点移除与对象销毁](arc-weak-and-cycle-references.md#节点移除与对象销毁是两件事)。

这里与上一节的 NLL 并不矛盾：`InFlightGuard` 实现了 `Drop`，销毁时还要通过 `&Worker`
把计数减一。即使源码中没有再次显式读取 `guard.worker`，编译器仍要保证该引用在 guard
销毁时有效。

`InFlightGuard` 与标准库的 `MutexGuard` 没有类型上的继承关系；两者只是都利用 `Drop`
自动完成收尾：前者减计数，后者解锁。计数器的原子操作见
[`AtomicUsize` 与内部可变性](atomic-usize-and-interior-mutability.md)，锁守卫见
[`Mutex`、`MutexGuard`、解引用与毒锁](mutex-guard-deref-and-poisoning.md)。

## 方法中的 `self`、`&self`、`&mut self`

方法接收者直接表达所有权需求：

```rust
struct Payment {
    status: String,
}

impl Payment {
    fn status(&self) -> &str {
        &self.status
    }

    fn capture(&mut self) {
        self.status = "CAPTURED".to_owned();
    }

    fn into_status(self) -> String {
        self.status
    }
}
```

| 接收者 | 含义 | 调用后原值 |
|---|---|---|
| `&self` | 只读借用 | 可以继续使用 |
| `&mut self` | 可变借用 | 借用结束后可以继续使用 |
| `self` | 取得所有权 | 通常不能再使用 |

```rust
let mut payment = Payment {
    status: "AUTHORIZED".to_owned(),
};

println!("{}", payment.status()); // &self
payment.capture();                 // &mut self

let status = payment.into_status(); // self，消费 payment
// payment 此后不能再使用
```

`into_` 开头的方法通常表示取得所有权并转换值，但这是一种命名惯例，不是语法规则。

`.into()` 则是 Into trait 的方法，消费接收者并返回目标类型；传入引用时，消费引用不等于
取得被引用对象的所有权。目标类型推断与 From / Into 的关系见
[类型转换专题](result-question-mark-and-error-propagation.md#类型转换intofromtryinto-与-tryfrom)。

## `Copy` 与 `Clone` 是 move 的两个特殊分支

实现 `Copy` 的小型值在原本会 move 的位置自动复制：

```rust
let first: u64 = 100;
let second = first;

println!("{first} {second}");
```

没有实现 `Copy` 的类型，如果双方确实都需要所有权，就显式 clone：

```rust
let first = String::from("hello");
let second = first.clone();

println!("{first} {second}");
```

不要把 `.clone()` 当成修复所有权错误的默认办法：

```rust
// 只读取，借用更合适。
fn validate(command: &String) {}

let command = String::from("capture");
validate(&command);
println!("{command}");
```

两者更详细的区别见[常见 `derive` 速查](common-derive.md)。

## 结构体字段也可以发生部分 move

```rust
#[derive(Debug)]
struct User {
    name: String,
    age: u32,
}

let user = User {
    name: "Alice".to_owned(),
    age: 20,
};

let name = user.name; // String 被 move
println!("{}", user.age); // age 是 Copy，仍可使用

// println!("{:?}", user); // 整个 user 已经不完整，不能再整体使用
```

这称为 partial move。常见解决方式是借用字段：

```rust
let name = &user.name;
println!("{name}");
println!("{}", user.age);
```

或者在确实要取走可选字段时使用 `Option::take()`：

```rust
struct Payment {
    channel_transaction_id: Option<String>,
}

let mut payment = Payment {
    channel_transaction_id: Some("tx-100".to_owned()),
};

let transaction_id = payment.channel_transaction_id.take();

assert_eq!(transaction_id.as_deref(), Some("tx-100"));
assert!(payment.channel_transaction_id.is_none());
```

`take()` 把 `Some(String)` 移出来，同时在原位置留下 `None`，所以结构体仍然是完整可用的值。

## 容器迭代时的所有权

三个常用迭代方式分别对应读取、修改和消费：

```rust
let mut names = vec![
    String::from("alice"),
    String::from("bob"),
];

for name in names.iter() {
    println!("{name}");       // name: &String
}

for name in names.iter_mut() {
    name.make_ascii_uppercase(); // name: &mut String
}

for name in names.into_iter() {
    println!("{name}");       // name: String
}

// names 已被 into_iter() 消费，不能再使用
```

| 写法 | 元素类型 | 容器之后能否使用 |
|---|---|---:|
| `iter()` | `&T` | 能 |
| `iter_mut()` | `&mut T` | 能，借用结束后 |
| `into_iter()` | `T` | 不能，容器已被消费 |

`for item in collection` 通常调用 `IntoIterator::into_iter(collection)`，也会消费拥有所有权的
容器；不想消费时写 `for item in &collection`。

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
[Arc：对象身份与值相等](arc-weak-and-cycle-references.md#ptr_eq对象身份与值相等)。

`Arc`、`Weak` 的生命周期关系见 [`Arc`、`Weak` 与循环引用](arc-weak-and-cycle-references.md)。
`Arc<Mutex<T>>` 的共享修改、锁守卫解引用与自动解锁见
[`Mutex`、`MutexGuard`、解引用与毒锁](mutex-guard-deref-and-poisoning.md)。

## 无状态支付请求中的完整例子

```rust
async fn capture_payment(
    command: CapturePaymentCommand,       // command 所有权 move 进函数
    repository: &PaymentRepository,       // 只借用 repository
) -> Result<PaymentCaptured, CaptureError> {
    let mut payment = repository
        .find(command.payment_id)          // PaymentId 是 Copy
        .await?;                           // 当前函数拥有 Payment

    let expected_version = payment.version();

    let event = payment.capture(
        &command.channel_transaction_id,   // 只借用 String 内容
    )?;

    repository
        .save(payment, expected_version)   // Payment move 给 repository
        .await?;

    Ok(event)                              // event move 给调用方
}
```

这段流程中的所有权变化是：

```text
调用方
  │ move command
  ▼
capture_payment
  │ borrow repository
  │ own + mutate payment
  │ move payment -> save
  │ move event
  ▼
调用方得到 PaymentCaptured
```

这是请求内的内存所有权；不同服务实例之间的支付一致性仍应由幂等键、事务、状态条件和版本号
保证，不能依靠 Rust 的 move 或 borrow。

## 常见编译错误

### `use of moved value`

```rust
let text = String::from("hello");
consume(text);
// consume(text); // value used here after move
```

检查真实意图：

```text
函数只读取？       -> 参数改为 &str 或 &T
原变量不再需要？   -> 保持 move
双方都需要所有权？ -> 明确调用 clone()
```

### `cannot borrow ... as mutable`

```rust
let text = String::from("hello");
// text.push('!'); // text 没有声明为 mut
```

变量需要声明为可变，并通过 `&mut` 传给修改者：

```rust
let mut text = String::from("hello");
text.push('!');
```

### `cannot borrow ... as mutable because it is also borrowed as immutable`

这表示一个共享引用后面仍会使用，却同时创建了可变引用。通常应缩短共享借用的使用范围，或者
调整代码顺序，不要立即用 `.clone()` 掩盖问题。

## 决策表

| 需求 | 参数或操作 |
|---|---|
| 函数只读取值 | `&T`；字符串常用 `&str` |
| 函数原地修改值 | `&mut T` |
| 函数接管值并负责后续处理 | `T`，发生 move |
| 函数消费后产生另一种值 | `self` / `T`，常用 `into_*` 命名 |
| 两个地方必须分别拥有值 | `.clone()` |
| 多处共享同一个不可变大对象 | `Arc<T>` |
| 小型无资源值允许隐式复制 | 实现 `Copy` |

## 一句话记忆

```text
move：这东西以后归你。
&T：借你看看，不能修改。
&mut T：暂时只借给你修改。
clone：我们各自拥有一份，具体复制多少由类型决定。
Copy：值很简单，move 时自动复制。
```
