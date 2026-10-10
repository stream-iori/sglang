# Rust：闭包、Fn 系列与条件等待

[返回索引](index.md) · [所有权基础](ownership-move-and-borrowing.md) · [Tokio 专题](tokio-basics-and-task-cancellation.md)

## 先理解一个调用

`my-smg` 的取消测试使用：

```rust
wait_until(Duration::from_secs(2), || worker.counter() == 1).await;
```

第二个参数不是已经算好的 bool，而是一段可再次执行的判断逻辑。

```text
传 bool：现在检查一次 → 等待函数只拿到 true 或 false
传闭包：保存检查办法 → 等待函数每轮重新读取计数
```

| 部分 | 含义 |
|---|---|
| `||` | 闭包参数列表为空，不是逻辑或 |
| `worker.counter() == 1` | 闭包的执行内容，返回 bool |
| 外部的 worker | 闭包捕获的变量，本例共享借用它 |
| `.await` | 等待 wait_until 返回的 Future，不是等待 bool |

有参数时写成 `|value| ...`，有多条语句时可以使用块：`|| { ... }`。
闭包是一个有具体类型的值；编译器为不同闭包表达式生成不同的匿名类型。

<a id="fn-family"></a>

## Fn、FnMut 与 FnOnce：区别在调用需要什么

| 能力 | 调用时的接收方式 | 常见行为 |
|---|---|---|
| `Fn` | `&self` | 只读取捕获状态，可重复调用 |
| `FnMut` | `&mut self` | 调用时可以修改捕获状态，可重复调用 |
| `FnOnce` | `self` | 消耗闭包，可以将捕获的非 Copy 值移出去 |

这些不是互斥标签：实现 Fn 的闭包也实现 FnMut 和 FnOnce；实现 FnMut 的闭包也实现 FnOnce。
FnOnce bound 只保证能按值调用一次，不表示所有满足它的闭包都只能调用一次。
参见 [标准库 FnMut](https://doc.rust-lang.org/std/ops/trait.FnMut.html)。

`wait_until` 的参数声明是：

```rust
mut condition: impl FnMut() -> bool
```

| 语法 | 职责 |
|---|---|
| `condition` | 参数变量名 |
| `mut` | 允许可变借用这个闭包值，以便调用 FnMut |
| `impl FnMut` | 参数接受一个满足约束的具体类型；不是 dyn 动态分发 |
| `()` | 调用闭包不传参数 |
| `-> bool` | 每次调用返回 bool |

用具名泛型表达，同一个约束可以写成 `F: FnMut() -> bool`，参数类型写 F。
泛型与 impl Trait 的基础见 [Trait 与分发](traits-dispatch-and-smart-pointers.md#impl-trait-与静态分发)。

计数检查闭包只读取 worker，可以满足 Fn；因此也能传给要求 FnMut 的接口。
选择 FnMut 是允许更广的重复判断场景，不是要求每次必须修改状态。
例如下面的闭包修改自己捕获的普通变量，需要 FnMut 调用能力：

```rust
let mut attempts = 0;
let mut condition = || {
    attempts += 1;
    attempts >= 3
};
assert!(!condition()); // 第 1 次
assert!(!condition()); // 第 2 次
assert!(condition());  // 第 3 次
```

本例为展示闭包能力，不是建议用尝试次数代替真实状态判断。
Fn 系列描述闭包的调用方式，不描述其内部调用是否通过原子变量、锁等修改其他对象。

### 三个小例子对照

```rust
// Fn：读捕获的变量，反复调用不会取走 String。
let name = String::from("worker-a");
let name_len = || name.len();
assert_eq!(name_len(), 8);
assert_eq!(name_len(), 8);
```

```rust
// FnMut：修改捕获的普通变量，因此调用闭包的绑定需要 mut。
let mut count = 0;
let mut increment = || {
    count += 1;
    count
};
assert_eq!(increment(), 1);
assert_eq!(increment(), 2);
```

```rust
// 只满足 FnOnce：把捕获的 String 返回给调用方，闭包不再保有它。
let name = String::from("worker-a");
let take_name = move || name;
let owned_name = take_name();
assert_eq!(owned_name, "worker-a");
// take_name(); // 编译错误：第一次调用已消费闭包。
```

| 例子 | 捕获值的变化 | 对应能力 |
|---|---|---|
| name_len | 保留，只读 | Fn，也满足 FnMut、FnOnce |
| increment | 保留，但可修改 | FnMut，也满足 FnOnce |
| take_name | String 在调用时移出 | 只满足 FnOnce |

Fn 的绑定不需要 mut；FnMut 调用需要能可变借用闭包。
这里“修改捕获变量”指第二个例子里普通整数的赋值，不是把所有带修改操作的闭包都归为 FnMut。

<a id="double-move"></a>

## 外层 move 闭包与内层 async move

延迟聊天服务器的路由使用：

```rust
post(move || async move {
    tokio::time::sleep(delay).await;
    "slow response"
})
```

这里有两个不同的值：负责接收调用的闭包，以及每次调用时创建的 Future。

```text
外部 delay
    ↓ 外层 move：按值捕获
路由处理闭包持有 delay
    ↓ 每次调用闭包，创建内层 async move
本次请求的 Future 持有 delay
    ↓ 被轮询后
等待时间，再返回响应
```

本例的 Duration 实现 Copy，内层按值捕获取得副本，不会把闭包中的 delay 永久取走。
所以同一个路由处理闭包可以重复调用。

不要推广成“加两个 move 就能重复使用任意值”：如果内层 Future 取走了外层闭包持有的
非 Copy String，而调用时没有另行 clone，外层闭包通常只能按 FnOnce 的方式调用。
`move` 决定捕获方式；Fn 系列能力还取决于调用时如何使用捕获值。
见 [Rust Reference：闭包能力与捕获](https://doc.rust-lang.org/reference/types/closure.html#call-traits-and-coercions)，
以及 [Tokio：async move 捕获规则](tokio-basics-and-task-cancellation.md#42-不写-move也可能移动变量)。

### 闭包与异步块不是同一种值

| 写法 | 创建什么 | 如何使用 |
|---|---|---|
| `move || { ... }` | 闭包 | 调用它，例如 handler() |
| `async move { ... }` | Future | 驱动它，例如 .await |

`post(move || async move { ... })` 中，post 的参数是闭包；闭包调用后返回 Future。
post 自己返回 MethodRouter，不是里面那个闭包。
判断依据是闭包语法，以及 Handler 的 Future 约束。
完整拆解见 [Axum：post、Handler 与 Future](axum-routing-handler-and-future.md)。

## 与条件等待及测试设计的关系

`wait_until` 负责循环、间隔和总等待期限；闭包负责判断“什么叫就绪”。
例如判断计数为 1、计数为 2，或健康状态已恢复，可以复用相同的等待机制。

闭包返回 false 时继续等待，返回 true 时结束；等待超时返回 Elapsed。
这个闭包是同步检查，不要在里面执行阻塞网络请求或试图直接使用 await。
如果检查本身需要异步操作，那是另一个接口需求，本例没有引入。

关联阅读：

- [Tokio：任务资源管理与等待条件](tokio-basics-and-task-cancellation.md#managed-task-resource-lifecycle)。
- [设计原则：机制与断言保持分离](desgin-principle.md#test-refactor-principles)。

## 一句话复习

```text
闭包传入的是“检查办法”，不是一次检查的结果。
move 管捕获；Fn、FnMut、FnOnce 管调用。
外层闭包可以被多次调用，每次创建一个独立的 Future。
```
