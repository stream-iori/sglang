# Rust：常见 `derive` 速查

## 一句话

`#[derive(...)]` 让编译器或过程宏根据类型的字段，自动生成 trait 实现：

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
struct WorkerId(String);
```

可以近似理解为编译器替你生成：

```rust
impl Debug for WorkerId { /* 按字段格式化 */ }
impl Clone for WorkerId { /* 逐字段 clone */ }
impl PartialEq for WorkerId { /* 逐字段比较 */ }
impl Eq for WorkerId {}
```

`derive` 只负责生成实现，不会改变字段的所有权、可见性或运行时生命周期。

## 两类 `derive`

### Rust 内置

无需添加依赖：

```text
Debug、Clone、Copy、Default
PartialEq、Eq、PartialOrd、Ord、Hash
```

### 第三方过程宏

由依赖 crate 提供，通常需要开启相应 feature：

```toml
[dependencies]
serde = { version = "1", features = ["derive"] }
thiserror = "2"
clap = { version = "4", features = ["derive"] }
```

```rust
use clap::Parser;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Parser)]
struct Args { /* ... */ }

#[derive(Serialize, Deserialize)]
struct Config { /* ... */ }

#[derive(Debug, Error)]
enum AppError { /* ... */ }
```

相同的 `#[derive(...)]` 语法，背后可能是编译器内置能力，也可能是 crate 导出的过程宏。

## 最常见组合

| 场景 | 常用组合 |
|---|---|
| 普通配置结构体 | `Debug, Clone, Default` |
| 值对象 / ID | `Debug, Clone, PartialEq, Eq, Hash` |
| 小型无资源状态枚举 | `Debug, Clone, Copy, PartialEq, Eq` |
| 可排序 key | `Debug, Clone, PartialEq, Eq, PartialOrd, Ord` |
| JSON / 配置数据 | `Debug, Clone, Serialize, Deserialize` |
| 应用错误 | `Debug, thiserror::Error` |
| CLI 参数 | `Debug, clap::Parser` |

不要把这张表当成固定模板。只派生代码真正需要的能力，尤其要谨慎对待 `Copy`、`Default`、
`Ord` 和面向外部协议的 `Serialize` / `Deserialize`。

## `Debug`：开发时查看值

```rust
#[derive(Debug)]
struct Request {
    id: u64,
    prompt: String,
}

let request = Request {
    id: 1,
    prompt: "hello".to_owned(),
};

println!("{request:?}");   // 紧凑格式
println!("{request:#?}");  // 美化格式
```

常用于日志、断言失败信息和临时排查。所有字段都必须实现 `Debug`。

`Debug` 面向开发者，不是稳定的用户协议；不要依赖其输出格式做解析，也不要无意间把 token、
密码等敏感字段写进日志。

## 理解 `Clone` / `Copy` 前，先理解 move

Rust 中，赋值、传参和返回值默认会发生 move（所有权转移）：

所有权、转移和借用的完整说明见[所有权、转移与借用](ownership-move-and-borrowing.md)。

```rust
let first = String::from("hello");
let second = first;

println!("{second}"); // 可以：String 现在由 second 拥有
// println!("{first}"); // 编译错误：first 的值已经被 move
```

这里没有复制字符串。可以把变化理解成：

```text
赋值前：first  ──拥有──> String("hello")

let second = first;

赋值后：first  ──已失效
        second ──拥有──> 原来的 String("hello")
```

Rust 这样做是为了保证 `String` 的堆内存只有一个所有者负责释放，避免两次释放同一块内存。
`Clone` 和 `Copy` 都能得到另一个值，但触发方式和适用类型不同。

## `Clone`：明确说“请再复制一份”

`Clone` 提供 `.clone()` 方法。调用者必须显式写出来：

```rust
let first = String::from("hello");
let second = first.clone();

println!("{first}");  // 可以
println!("{second}"); // 也可以
```

对于 `String`，`.clone()` 会申请新的堆内存并复制字符：

```text
first  ──拥有──> String("hello")，堆内存 A
second ──拥有──> String("hello")，堆内存 B
```

两个值互相独立：

```rust
let first = String::from("hello");
let mut second = first.clone();
second.push_str(" rust");

assert_eq!(first, "hello");
assert_eq!(second, "hello rust");
```

### `#[derive(Clone)]` 做了什么

```rust
#[derive(Clone, Debug)]
struct Config {
    endpoint: String,
    retries: u32,
}
```

派生实现近似于：

```rust
impl Clone for Config {
    fn clone(&self) -> Self {
        Self {
            endpoint: self.endpoint.clone(),
            retries: self.retries.clone(),
        }
    }
}
```

也就是逐字段调用 `.clone()`。只要有一个字段没有实现 `Clone`，整个结构体就不能自动派生
`Clone`。

### `clone()` 不一定是深拷贝

复制的具体含义由字段类型决定：

| 类型 | `.clone()` 通常做什么 |
|---|---|
| `String` | 申请新内存，复制字符串内容 |
| `Vec<T>` | 申请新内存，逐个 clone 元素 |
| `u32`、`bool` | 直接复制值 |
| `Arc<T>` | 只增加强引用计数，两个 `Arc` 仍指向同一个 `T` |

```rust
use std::sync::Arc;

let first = Arc::new(String::from("hello"));
let second = first.clone();

assert!(Arc::ptr_eq(&first, &second)); // 指向同一个 String
assert_eq!(Arc::strong_count(&first), 2);
```

所以 `Clone` 只表示“可以显式得到另一个值”，不保证复制便宜，也不保证底层数据完全独立。

## `Copy`：使用旧值时自动复制

先看整数：

```rust
let first: u32 = 10;
let second = first;

println!("{first}");  // 可以
println!("{second}"); // 可以
```

这段代码与 `String` 的例子看起来相同，但 `u32` 实现了 `Copy`。所以 `let second = first`
不会让 `first` 失效，而是自动复制一个值：

```text
赋值前：first = 10
赋值后：first = 10，second = 10
```

`Copy` 没有一个需要主动调用的 `.copy()` 方法。它是一个标记，告诉编译器：这个类型可以在
原本会 move 的地方自动复制。

### 给自定义类型派生 `Copy`

```rust
#[derive(Clone, Copy, Debug)]
struct Point {
    x: i32,
    y: i32,
}

let first = Point { x: 10, y: 20 };
let second = first;

println!("{first:?}");  // 可以，发生了 Copy
println!("{second:?}");
```

`i32` 是 `Copy`，所以只包含 `i32` 的 `Point` 也可以派生 `Copy`。

下面这个类型则不能派生 `Copy`：

```rust
// ❌ 这段代码无法编译
#[derive(Clone, Copy)]
struct User {
    name: String, // String 不是 Copy
}
```

如果允许按位复制 `String`，两个值就会保存相同的堆指针，并在离开作用域时尝试释放同一块内存。
因此，拥有堆内存、文件句柄或引用计数等资源的类型通常不能实现 `Copy`。

### 传参时也会触发 move 或 Copy

```rust
fn print_number(value: u32) {
    println!("{value}");
}

let number = 10;
print_number(number); // number 被 Copy 进函数
print_number(number); // 仍然可以使用
```

对没有实现 `Copy` 的 `String`，第一次按值传参后原变量就不能再使用，除非传引用或显式
`.clone()`：

```rust
fn print_text(value: &str) {
    println!("{value}");
}

let text = String::from("hello");
print_text(&text); // 借用，没有 move，也没有复制字符串
print_text(&text); // 可以继续借用
```

### `Clone` 与 `Copy` 对照

| 问题 | `Clone` | `Copy` |
|---|---|---|
| 如何触发 | 显式调用 `.clone()` | 赋值、传参等位置隐式发生 |
| 是否可能昂贵 | 可能 | 应当是简单、便宜的值复制 |
| 能否用于拥有资源的类型 | 可以，例如 `String`、`Vec<T>`、`Arc<T>` | 通常不可以 |
| 原变量能否继续使用 | 调用 `.clone()` 后可以 | 自动可以 |
| trait 关系 | 可以单独实现 | 必须同时实现 `Clone` |

判断过程：

```text
只想借来使用，不需要新值？ -> 传 &T / &mut T
需要明确产生另一个值？     -> Clone
类型是简单小值，隐式复制符合直觉？ -> Clone + Copy
```

`Copy` 的硬性约束包括：所有字段都是 `Copy`、类型同时实现 `Clone`，并且类型没有实现
`Drop`。常见的 `Copy` 类型有整数、浮点数、`bool`、`char`、共享引用 `&T`，以及只包含这些
字段的小型结构体或枚举。

## 无状态支付微服务中的 `Clone` / `Copy`

在 DDD 支付系统中，判断是否实现 `Clone` / `Copy` 时，需要分开看两个问题：

```text
Rust 所有权问题：
当前进程、当前请求中，一个值如何传递和复制？

分布式一致性问题：
不同实例、请求和消息之间，支付状态如何保持正确？
```

`Clone` / `Copy` 只回答第一个问题。第二个问题通常由数据库事务、状态条件、乐观锁、幂等键、
Outbox 和消息去重解决。

### “无状态”不等于没有业务状态

无状态服务不在某个进程中长期保存权威业务状态。一次请求通常经历：

```text
请求到达
  -> 从数据库或消息重建当前状态
  -> 在内存中短暂构造领域对象
  -> 执行业务规则
  -> 条件更新数据库并写入 Outbox
  -> 请求结束，内存对象释放
```

持久状态仍然存在，只是位于进程外：

| 位置 | 典型状态 |
|---|---|
| 数据库 | 支付状态、金额、版本号 |
| Redis | 幂等记录、短期缓存 |
| 消息队列 | Command、Domain Event |
| Outbox | 已提交但尚未发布的事件 |

所以同一笔支付天然可能在不同实例中出现多个内存副本：

```text
实例 A：Payment(id=100, status=Authorized, version=7)
实例 B：Payment(id=100, status=Authorized, version=7)
```

即使 `Payment` 没有实现 `Clone`，两个实例仍能分别从数据库加载出这两个值。因此：

```text
禁止 Clone != 防止分布式并发
```

### `Copy`：适合小型、固定大小的值对象

支付 ID 是典型例子：

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct PaymentId(u128);

fn load_payment(id: PaymentId) {}
fn record_metric(id: PaymentId) {}

let id = PaymentId(100);
load_payment(id);
record_metric(id); // id 自动 Copy，仍可继续使用
```

复制 `PaymentId` 只是复制身份值，不是创建一笔新支付：

```text
复制 PaymentId(100) -> 得到相同的编号值
创建新 Payment      -> 应由领域流程产生新的唯一 ID
```

固定大小的金额和状态枚举也可以考虑 `Copy`：

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Currency {
    Cny,
    Usd,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Money {
    minor_units: i64,
    currency: Currency,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PaymentStatus {
    Created,
    Authorized,
    Captured,
    Refunded,
    Failed,
}
```

这里的判断依据不是“它们属于 DDD Value Object”，而是：

- 复制在业务语义上合理；
- 类型很小且大小固定；
- 所有字段都是 `Copy`；
- 没有堆内存或其他资源所有权。

值对象不一定能 `Copy`。例如订单号包含 `String`，只能按需 `Clone`：

```rust
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct MerchantOrderNo(String);
```

```text
DDD 分类回答：这个值在业务上是否允许复制？
Rust 类型回答：它在技术上能否安全地隐式复制？
```

### Request、Command 和 DTO：一个消费者优先 move

```rust
#[derive(Clone, Debug)]
struct CapturePaymentCommand {
    payment_id: PaymentId,
    idempotency_key: String,
    operator: String,
}
```

如果只有一个消费者，直接转移所有权，不需要 clone：

```rust
async fn handler(command: CapturePaymentCommand) {
    payment_service.capture(command).await;
    // command 已经 move 给 capture
}
```

多个组件确实都需要拥有数据时，再显式 clone：

```rust
async fn handler(command: CapturePaymentCommand) {
    audit_service.record(command.clone()).await;
    payment_service.capture(command).await;
}
```

如果审计只需要幂等键，可以只复制那个字段：

```rust
async fn handler(command: CapturePaymentCommand) {
    let idempotency_key = command.idempotency_key.clone();

    payment_service.capture(command).await;
    audit_service.record_key(idempotency_key).await;
}
```

选择顺序是：

```text
一个消费者  -> move
只是读取    -> &T
多个所有者  -> clone()
进程内共享大型不可变值 -> Arc<T>
```

### Entity / Aggregate 可以 `Clone`，但它只产生内存快照

无状态微服务中，不能简单规定 Entity 或 Aggregate 永远不能实现 `Clone`。例如为了记录状态转换
前后的差异，可以明确复制一个请求级快照：

```rust
#[derive(Clone, Debug)]
struct Payment {
    id: PaymentId,
    status: PaymentStatus,
    version: u64,
    channel_transaction_id: Option<String>,
}

let payment = repository.load(payment_id).await?;

let before = payment.clone();
let mut after = payment;

let event = after.capture()?;
audit.record_transition(&before, &after).await?;
repository.save_with_version(after, before.version, event).await?;
```

这里的两个值表示：

```text
before -> 状态转换前的内存快照
after  -> 状态转换后的内存快照
```

它们不是两笔新支付，也不代表数据库允许两个状态同时写入。

是否为 Aggregate 派生 `Clone` 仍然要谨慎：自动派生会复制全部字段，容易产生不必要的大对象
复制，也可能让单次请求中的状态分支变得难以追踪。如果只需要查询或审计数据，可以定义专门的
只读快照：

```rust
#[derive(Clone, Debug)]
struct PaymentSnapshot {
    id: PaymentId,
    status: PaymentStatus,
    version: u64,
    channel_transaction_id: Option<String>,
}

impl From<&Payment> for PaymentSnapshot {
    fn from(payment: &Payment) -> Self {
        Self {
            id: payment.id,
            status: payment.status,
            version: payment.version,
            channel_transaction_id: payment.channel_transaction_id.clone(),
        }
    }
}
```

```text
Payment         -> 有行为、负责维护业务不变量的请求级聚合
PaymentSnapshot -> 用于查询、审计或传输的只读数据
```

不为 Aggregate 实现 `Clone` 可以作为防止请求内误用的 API 设计，但不能替代分布式一致性机制。

### 支付一致性依靠版本条件，而不是禁止 `Clone`

假设两个实例同时加载版本 7：

```text
实例 A：准备 Capture
实例 B：准备 Refund
```

持久化时需要使用状态条件或乐观锁：

```sql
UPDATE payments
SET status = 'CAPTURED',
    version = version + 1
WHERE payment_id = ?
  AND status = 'AUTHORIZED'
  AND version = 7;
```

```text
affected_rows == 1 -> 更新成功
affected_rows == 0 -> 状态或版本已变化，拒绝或重新加载后重试
```

典型支付流程是：

```text
检查幂等键
  -> 加载 Payment + version
  -> 执行领域状态机
  -> 使用 version 做条件更新
  -> 同一数据库事务写 Outbox
  -> 消费端使用 Inbox / event_id 去重
```

无论当前请求有没有调用 `.clone()`，最终写入都必须经过这条一致性边界。

### Domain Event：按分发方式选择 `Clone` 或 `Arc`

领域事件是已经发生的不可变事实，常常需要交给多个进程内处理器：

```rust
#[derive(Clone, Debug)]
struct PaymentCaptured {
    payment_id: PaymentId,
    amount: Money,
    channel_transaction_id: String,
}
```

事件较小时可以 clone：

```rust
message_handler.handle(event.clone()).await?;
audit_handler.handle(event).await?;
```

进程内广播大型事件时，可以共享同一个不可变值：

```rust
use std::sync::Arc;

let event = Arc::new(event);

handler_a.handle(Arc::clone(&event)).await?;
handler_b.handle(Arc::clone(&event)).await?;
```

`Arc` 只解决当前进程的所有权共享。事件一旦通过 Kafka 等消息系统跨进程发送，仍然要经过
序列化、持久化和消费者去重，不能依靠 `Arc`。

### Repository client 与数据库 transaction 不一样

Repository client 内部如果只是连接池句柄，通常可以廉价 Clone：

```text
PaymentRepository
  -> clone 连接池句柄
  -> 多个请求共享底层连接池
```

但数据库 transaction 有明确的 commit / rollback 生命周期，通常不应实现 `Clone`：

```rust
async fn save_payment(
    tx: &mut PaymentTransaction,
    payment: &Payment,
) -> Result<(), RepositoryError> {
    // 使用同一个事务执行更新和写 Outbox
}
```

如果 transaction 可以随意复制，就会产生“谁提交、谁回滚、是否仍是同一事务”等语义歧义。

### 无状态支付服务中的推荐选择

| 类型 | 推荐 | 原因 |
|---|---|---|
| `PaymentId(u128)` | `Copy + Clone` | 小型、固定大小、复制无歧义 |
| `PaymentStatus` | `Copy + Clone` | 小型无资源枚举 |
| 固定大小的 `Money` | 可考虑 `Copy + Clone` | 取决于内部表示 |
| `MerchantOrderNo(String)` | `Clone` | 业务上可复制，但有堆内存成本 |
| HTTP / gRPC DTO | 按需 `Clone` | 单消费者优先 move |
| Command | 按需 `Clone` | 重试或多消费者确实需要时使用 |
| Domain Event | `Clone` 或 `Arc<Event>` | 取决于大小和进程内分发方式 |
| `Payment` Aggregate | 可以 `Clone`，但不应默认滥用 | Clone 只产生内存快照 |
| Query Snapshot / Read Model | 通常可以 `Clone` | 本身就是数据副本 |
| Repository client | 通常可以 `Clone` | 常见实现只是复制连接池句柄 |
| 数据库 transaction | 通常不能 `Clone` | 具有明确事务生命周期 |

最终决策可以简化为：

```text
复制在业务语义上不合理？
  -> 不实现 Clone / Copy

只需要临时读取？
  -> 使用 &T，不复制

值很小、固定大小、没有资源，隐式复制符合直觉？
  -> Copy + Clone

允许复制，但复制可能分配内存或共享底层资源？
  -> Clone，并让调用方显式调用 clone()

需要跨实例保证支付状态正确？
  -> 乐观锁 + 幂等键 + 事务 + Outbox / Inbox
     Clone / Copy 不负责解决这个问题
```

## `Default`：提供默认值

```rust
#[derive(Debug, Default)]
struct RetryConfig {
    max_retries: usize, // 0
    enabled: bool,      // false
    endpoint: String,   // ""
}

let config = RetryConfig::default();
```

结构体的派生实现逐字段调用 `Default::default()`。如果业务默认值不是各字段的零值，应手写：

```rust
impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            enabled: true,
            endpoint: "http://127.0.0.1:3000".to_owned(),
        }
    }
}
```

枚举需要用 `#[default]` 指定默认的 unit variant：

```rust
#[derive(Debug, Default)]
enum Mode {
    #[default]
    Auto,
    Manual,
}
```

能派生不代表语义合理。如果“不提供值”应该报错，就不应为了方便构造而添加 `Default`。

## `PartialEq`：让类型可以使用 `==` 和 `!=`

没有实现 `PartialEq` 的自定义类型，不能直接使用 `==`：

```rust
// ❌ 这段代码无法编译
struct ModelKey {
    name: String,
    revision: String,
}

let left = ModelKey {
    name: "llama".to_owned(),
    revision: "v1".to_owned(),
};
let right = ModelKey {
    name: "llama".to_owned(),
    revision: "v1".to_owned(),
};

// 编译错误：ModelKey 没有实现 PartialEq
assert!(left == right);
```

派生 `PartialEq` 后就可以比较：

```rust
#[derive(Debug, PartialEq)]
struct ModelKey {
    name: String,
    revision: String,
}

let left = ModelKey {
    name: "llama".to_owned(),
    revision: "v1".to_owned(),
};
let same = ModelKey {
    name: "llama".to_owned(),
    revision: "v1".to_owned(),
};
let other_revision = ModelKey {
    name: "llama".to_owned(),
    revision: "v2".to_owned(),
};

assert!(left == same);
assert!(left != other_revision);
assert_eq!(left, same);
```

对于结构体，自动派生的规则是“所有对应字段都相等，两个值才相等”：

```text
left.name == right.name
    并且
left.revision == right.revision
    才能得到
left == right
```

近似于编译器生成：

```rust
impl PartialEq for ModelKey {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && self.revision == other.revision
    }
}
```

因此所有参与比较的字段也必须实现 `PartialEq`。`assert_eq!` 需要 `PartialEq` 来比较，并需要
`Debug` 在断言失败时打印左右值。

### enum 如何比较

```rust
#[derive(Debug, PartialEq)]
enum State {
    Waiting,
    Running { worker_id: u32 },
}

assert_eq!(State::Waiting, State::Waiting);
assert_eq!(
    State::Running { worker_id: 7 },
    State::Running { worker_id: 7 },
);
assert_ne!(
    State::Running { worker_id: 7 },
    State::Running { worker_id: 8 },
);
assert_ne!(State::Waiting, State::Running { worker_id: 7 });
```

两个 enum 值只有在 variant 相同、variant 内所有字段也相同时才相等。

## `Eq`：保证“每个值都等于自己”

`Eq` 容易让人困惑，是因为它不提供新的 `==` 方法。真正实现比较的是 `PartialEq`；`Eq` 是在
此基础上的一个保证：该类型的相等关系是完整的等价关系。

最直观的一条保证是：

```text
对任意值 x，都有 x == x
```

因此 `Eq` 依赖 `PartialEq`，一般一起派生：

```rust
#[derive(Debug, PartialEq, Eq)]
struct RequestId(u64);
```

可以这样记住两者的分工：

```text
PartialEq：实现“怎么比较”，提供 == 和 !=
Eq：额外承诺“这种比较没有例外，每个值都等于自己”
```

### 为什么浮点数只有 `PartialEq`，没有 `Eq`

浮点数有一个特殊值 `NaN`：

```rust
let value = f64::NAN;

assert!(value != value);
```

`NaN` 不等于它自己，违反了 `Eq` 的保证。所以：

```text
f64: PartialEq  ✓  可以写 1.0 == 1.0
f64: Eq         ✗  因为 NaN != NaN
```

包含 `f64` 字段的结构体可以派生 `PartialEq`，但不能派生 `Eq`：

```rust
#[derive(Debug, PartialEq)]
struct Score {
    value: f64,
}

// #[derive(Eq)] 会编译失败，因为 f64 没有实现 Eq。
```

这就是名字里 `Partial` 的核心含义：比较操作存在，但它不一定满足完整等价关系的全部规则。
这里的 `PartialEq::eq()` 仍然返回 `bool`，不是 `Option<bool>`。

### 为什么 `HashMap` 的 key 常要求 `Eq + Hash`

```rust
use std::collections::HashMap;

#[derive(Debug, PartialEq, Eq, Hash)]
struct RequestId(u64);

let mut requests = HashMap::new();
requests.insert(RequestId(7), "running");

assert_eq!(requests.get(&RequestId(7)), Some(&"running"));
```

哈希表需要确信 key 的相等关系没有 `NaN` 这种例外，并要求相等的值产生相同哈希结果，所以 key
通常同时实现 `PartialEq + Eq + Hash`。

### 派生比较是否应该包含所有字段

自动派生会比较所有字段：

```rust
#[derive(Debug, PartialEq, Eq)]
struct Worker {
    id: u64,
    completed_requests: u64,
}
```

即使两个 worker 的 `id` 相同，只要统计值不同，它们也会被判定为不相等。如果业务身份只由
`id` 决定，可以拆出单独的 `WorkerId` 类型，或者谨慎地手写 `PartialEq`：

```rust
#[derive(Debug)]
struct Worker {
    id: u64,
    completed_requests: u64,
}

impl PartialEq for Worker {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Worker {}
```

如果该类型还实现 `Hash`，哈希也必须只使用相同的身份字段，否则会破坏哈希表规则。

### `PartialEq` 与 `Eq` 对照

| 类型 | `PartialEq` | `Eq` | 原因 |
|---|---:|---:|---|
| `u64`、`bool`、`String` | 是 | 是 | 每个值都等于自己 |
| `f32`、`f64` | 是 | 否 | `NaN != NaN` |
| 字段全是 `Eq` 的结构体 | 可派生 | 可派生 | 逐字段比较满足完整等价关系 |
| 含 `f64` 的结构体 | 可派生 | 不可派生 | 内部浮点值可能是 `NaN` |

## `PartialOrd` 与 `Ord`：排序

```rust
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Priority(u32);
```

结构体按字段声明顺序做字典序比较；枚举先按 variant 的声明顺序比较，再比较 variant 内字段。

```text
PartialOrd -> 可能无法比较，partial_cmp 返回 Option<Ordering>
Ord        -> 任意两个值都有确定顺序，cmp 返回 Ordering
```

因为浮点数存在 `NaN`，所以它们只有 `PartialOrd`，没有 `Ord`。

派生顺序未必等于业务顺序。例如版本号字符串按字典序比较时，`"10" < "2"`；调度优先级也可能
需要反向或多字段规则。这些情况应手写比较或使用合适的领域类型。

## `Hash`：作为哈希集合的 key

```rust
use std::collections::HashSet;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct WorkerKey {
    model: String,
    address: String,
}

let mut workers = HashSet::new();
workers.insert(WorkerKey {
    model: "demo".to_owned(),
    address: "127.0.0.1:3000".to_owned(),
});
```

`HashMap` / `HashSet` 的 key 通常需要 `Eq + Hash`。两者必须保持一致：

```text
a == b  =>  hash(a) == hash(b)
```

同时派生 `PartialEq, Eq, Hash` 时，编译器会按相同字段生成实现，最不容易破坏这条约束。若手写
其中一个，必须确认另一个使用完全一致的身份字段。

## Serde：`Serialize` 与 `Deserialize`

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
struct WorkerConfig {
    model_id: String,

    #[serde(default)]
    enabled: bool,

    #[serde(skip_serializing_if = "Option::is_none")]
    api_key: Option<String>,
}
```

```text
Serialize：Rust 值 -> JSON / YAML / MessagePack 等数据模型
Deserialize：外部数据 -> Rust 值
```

常用辅助属性：

| 属性 | 作用 |
|---|---|
| `#[serde(rename = "x")]` | 修改单个字段或 variant 的外部名称 |
| `#[serde(rename_all = "snake_case")]` | 批量修改命名规则 |
| `#[serde(default)]` | 字段缺失时使用 `Default` |
| `#[serde(skip)]` | 序列化和反序列化都跳过 |
| `#[serde(skip_serializing_if = "...")]` | 满足条件时不输出字段 |
| `#[serde(flatten)]` | 将嵌套结构展开到当前对象 |

派生 Serde trait 相当于定义外部数据协议。重命名字段、修改 enum 表示形式或增加必填字段，都可能
是兼容性变更。

SGLang 中的配置类型大量使用这一组合，见
[`config/types.rs`](../../../sgl-model-gateway/src/config/types.rs)。

## thiserror：`Error`

`thiserror` 没有发明新的错误体系，它只是自动生成标准库要求的 trait 实现。下面两种写法
功能相同。

如果还不熟悉 `fmt::Formatter<'_>`、生命周期、`Display`、`Error::source()` 或 `dyn Error`，先读
[`Formatter<'_>`、生命周期、`Display` 与 `Error`](formatter-lifetimes-display-and-error.md)。

### 只使用 Rust 标准库

```rust
use std::{error::Error, fmt, io};

#[derive(Debug)]
enum CaptureError {
    InvalidStatus(u64),
    Repository(io::Error),
}

impl fmt::Display for CaptureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidStatus(payment_id) => {
                write!(f, "payment {payment_id} has invalid status")
            }
            Self::Repository(_) => write!(f, "repository operation failed"),
        }
    }
}

impl Error for CaptureError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Repository(error) => Some(error),
            Self::InvalidStatus(_) => None,
        }
    }
}

impl From<io::Error> for CaptureError {
    fn from(error: io::Error) -> Self {
        Self::Repository(error)
    }
}
```

除了错误 enum，标准库写法还要手写：

```text
Display      -> 错误显示什么文字
Error        -> 哪个字段是下层错误 source
From         -> ? 如何把下层错误转换成当前错误
```

### 使用 `thiserror` 的等价写法

```rust
use thiserror::Error;

#[derive(Debug, Error)]
enum CaptureError {
    #[error("payment {0} has invalid status")]
    InvalidStatus(u64),

    #[error("repository operation failed")]
    Repository(#[from] std::io::Error),
}
```

两段代码的对照关系是：

| 标准库手写 | `thiserror` |
|---|---|
| `impl fmt::Display` | `#[error("...")]` |
| `impl Error { source(...) }` | 自动把 `#[from]` 字段作为 source |
| `impl From<io::Error>` | `#[from]` |

两种写法都能让 `?` 自动转换错误：

```rust
fn query_payment() -> Result<(), std::io::Error> {
    Err(std::io::Error::other("database unavailable"))
}

fn capture_payment() -> Result<(), CaptureError> {
    query_payment()?; // io::Error -> CaptureError::Repository
    Ok(())
}
```

这行 `?` 近似于：

```rust
query_payment().map_err(CaptureError::Repository)?;
```

因此 `thiserror` 的优势不是功能更多，而是把三个实现压缩成两个属性：`#[error("...")]` 生成
`Display`，`#[from]` 生成错误转换并把内部错误标记为 source。新增 variant 时不用同步维护多段
`match`。

`thiserror` 在编译期生成的仍然是标准 `std::error::Error` 实现，调用方不需要依赖或感知
`thiserror`。它不会生成 `Debug`，所以通常同时写 `#[derive(Debug, Error)]`。实际用法见
[`core/error.rs`](../../../sgl-model-gateway/src/core/error.rs) 和
[`utils/error.rs`](../../../rust/sglang-server/src/utils/error.rs)。

## clap：`Parser`、`Args`、`Subcommand`、`ValueEnum`

```rust
use clap::{Parser, ValueEnum};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum LogFormat {
    Text,
    Json,
}

#[derive(Debug, Parser)]
struct Cli {
    #[arg(long, default_value_t = 3000)]
    port: u16,

    #[arg(long, value_enum, default_value_t = LogFormat::Text)]
    log_format: LogFormat,
}
```

| Derive | 用途 |
|---|---|
| `Parser` | 定义顶层命令行解析器 |
| `Args` | 定义一组可复用参数 |
| `Subcommand` | 用 enum 定义子命令 |
| `ValueEnum` | 将有限枚举映射为命令行参数值 |

SGL Model Gateway 的命令行入口同时使用了这些派生，见
[`main.rs`](../../../sgl-model-gateway/src/main.rs)。

## 泛型类型的 trait bound

自动派生会根据字段生成所需约束：

```rust
#[derive(Debug, Clone)]
struct Envelope<T> {
    value: T,
}
```

生成的实现近似为：

```rust
impl<T: Debug> Debug for Envelope<T> { /* ... */ }
impl<T: Clone> Clone for Envelope<T> { /* ... */ }
```

所以 `Envelope<NotClone>` 类型本身仍然可以存在，只是不能调用 `.clone()`。如果宏推导的约束
过强或不符合语义，可以手写 trait 实现；Serde 等第三方宏也通常提供属性来覆盖 bound。

## 常见编译错误怎么看

### 某个字段没有实现目标 trait

```text
the trait bound `SomeField: Clone` is not satisfied
```

派生实现会递归要求相关字段实现该 trait。可选做法是：

- 给字段类型也实现或派生该 trait；
- 删除不需要的 derive；
- 改用 `Arc<T>` 等符合所有权语义的类型；
- 手写实现，只处理真正属于该 trait 语义的字段。

### 找不到第三方 derive 宏

```text
cannot find derive macro `Serialize` in this scope
```

依次检查：

1. `Cargo.toml` 是否添加依赖并开启 `derive` feature；
2. 是否 `use serde::Serialize;`；
3. 或者改用完整路径 `#[derive(serde::Serialize)]`。

### `Eq`、`Ord` 被浮点字段阻止

这通常不是“少写了一个 derive”，而是浮点数本身不具备完全相等或全序语义。应先定义业务上
如何处理 `NaN`，再选择包装类型或手写比较逻辑。

## 派生前的检查清单

```text
Debug       输出是否可能泄露敏感数据？
Clone       复制成本是否可以接受？
Copy        隐式复制是否符合类型的资源语义？
Default     零值是否真的是合法业务默认值？
Eq / Hash   哪些字段定义对象身份？两者是否一致？
Ord         字段声明顺序是否就是业务排序顺序？
Serde       这是否会形成需要长期兼容的外部协议？
Error       source 和自动 From 转换是否准确？
```

## 一句话记忆

```text
Debug 看值，Clone 显式复制，Copy 隐式复制，Default 构造默认值；
Eq 判断相等，Ord 决定顺序，Hash 支持哈希 key；
Serde 定义数据协议，thiserror 定义错误，clap 定义命令行。
```
