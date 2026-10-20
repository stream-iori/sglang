# 共享借用与可变借用

[分类目录](index.md) · [总入口](../../index.md) · [问题索引](../../questions.md)

**借用只提供临时访问，不接管数据；共享读与独占修改不能冲突。**

| 阅读准备 | 内容 |
|---|---|
| 前置 | [struct、enum、impl 与 Self](../values/struct-enum-and-impl.md) |
| 源码入口 | [src/policy.rs](../../../../src/policy.rs)：`Policy::select` |
| 阅读时机 | 按当前课程需要阅读，不据此跳过 learning 的阶段验收。 |

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

## 多个共享借用，或者一个可变借用

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

## `cannot borrow ... as mutable`

```rust
let text = String::from("hello");
// text.push('!'); // text 没有声明为 mut
```

变量需要声明为可变，并通过 `&mut` 传给修改者：

```rust
let mut text = String::from("hello");
text.push('!');
```

## `cannot borrow ... as mutable because it is also borrowed as immutable`

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

## 不能把 `&` 传给要求 `&mut` 的参数

trait 契约要求 `&mut self` 时，变量绑定和传参都必须允许可变借用：

```rust
let mut policy = RoundRobin::new();
select_with_policy(&mut policy, &workers);
```

## 自查与关联

为什么 policy 是 &mut，而 workers 是共享切片？

[Move 与部分移动](move-and-partial-move.md) · [self、mut self、&self 与 &mut self](method-receivers.md) · [Copy 与 Clone：复制值还是共享句柄](copy-and-clone.md)
