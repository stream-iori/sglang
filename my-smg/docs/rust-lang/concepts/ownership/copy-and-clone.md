# Copy 与 Clone：复制值还是共享句柄

[分类目录](index.md) · [总入口](../../index.md) · [问题索引](../../questions.md)

**Copy 是隐式复制能力，Clone 是显式请求；clone 是否复制底层内容取决于类型。**

| 阅读准备 | 内容 |
|---|---|
| 前置 | [Move 与部分移动](move-and-partial-move.md)基础部分；项目应用另按节标题补前置。 |
| 源码入口 | [src/circuit_breaker.rs](../../../../src/circuit_breaker.rs)：`CircuitState` |
| 阅读时机 | T06 读到基础自查；派生在阶段 3、Arc 在阶段 7 回查。 |

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

## 传参时也会触发 move 或 Copy

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

## `Clone` 与 `Copy` 对照

| 问题 | `Clone` | `Copy` |
|---|---|---|
| 如何触发 | 显式调用 `.clone()` | 赋值、传参等位置隐式发生 |
| 是否可能昂贵 | 可能 | 应当是简单、便宜的值复制 |
| 能否用于拥有资源的类型 | 可以，例如 `String`、`Vec<T>` | 通常不可以 |
| 原变量能否继续使用 | 调用 `.clone()` 后可以 | 自动可以 |
| 能力关系 | 可以单独实现 | 必须同时实现 `Clone` |

判断过程：

```text
只想借来使用，不需要新值？ -> 传 &T / &mut T
需要明确产生另一个值？     -> Clone
类型是简单小值，隐式复制符合直觉？ -> Clone + Copy
```

`Copy` 的硬性约束包括：所有字段都是 `Copy`、类型同时实现 `Clone`，并且类型没有实现
`Drop`。常见的 `Copy` 类型有整数、浮点数、`bool`、`char`、共享引用 `&T`，以及只包含这些
字段的小型结构体或枚举。

含 String 字段的结构体不能派生 Copy，因为字段不全是 Copy；后文再解释派生。

## 基础自查

| 类型 | 问题 |
|---|---|
| 预测 | u32 赋值后原变量还能用吗？String 不 clone 时呢？ |
| 解释 | 为什么 String::clone 得到独立内容，而普通赋值不复制文本？ |
| 边界 | 一个含 String 的结构体能靠派生 Copy 允许隐式复制吗？ |

先写下判断与原因，再用自己已有的代码验证；遇到疑问回查本篇，不需要提前读后续应用。

## 阶段 3：派生自定义类型能力

先读[trait](../types/traits-and-bounds.md)、[derive](../types/derive-and-default.md)。

## `#[derive(Clone)]` 做了什么

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

## 给自定义类型派生 `Copy`

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

## 阶段 7：共享句柄的 Clone

先读[Arc](../sharing/rc-and-arc.md)。

## `clone()` 不一定是深拷贝

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
示例中的 `Arc::ptr_eq` 检查对象身份，不是内容相等，详见
[Arc：对象身份与值相等](../../walkthroughs/circuit-permit.md)。


综合题：Arc::clone 与 String::clone 都会复制底层文本吗？
