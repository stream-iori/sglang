# Option、Result 与容器中的借用和取值

[分类目录](index.md) · [总入口](../../index.md) · [问题索引](../../questions.md)

**Option 表示有无，Result 表示成败；as_ref 和 as_deref 借用，take 取走拥有值。**

| 阅读准备 | 内容 |
|---|---|
| 前置 | [struct、enum、impl 与 Self](../values/struct-enum-and-impl.md) · [共享借用与可变借用](../ownership/borrowing.md) |
| 源码入口 | [src/gateway/tests/support.rs](../../../../src/gateway/tests/support.rs)：`handle: Option<JoinHandle<T>>` |
| 阅读时机 | 按当前课程需要阅读，不据此跳过 learning 的阶段验收。 |

## 先分清两种容器

| 类型 | 分支 | 项目用途 |
|---|---|---|
| Option<T> | Some(T) / None | 句柄可能存在，也可能已经取走 |
| Result<T, E> | Ok(T) / Err(E) | 操作成功得到值，失败保留原因 |

例如 ManagedTask 的字段是 Option<JoinHandle<T>>，Some 表示还拥有句柄，take 后用 None 表示字段已清空。None 不等于 Err：它本身不带失败原因。

接着使用独立的 Option<Box<i32>> 例子，只比较借用与取值，不启动线程或异步任务。


## 用 Box 统一理解 `as_ref()`、`as_deref()` 与 `take()`

**`as_ref()` 借用盒子，`as_deref()` 借用盒子指向的整数，`take()` 把盒子取走。**
这里三个方法都调用在 `Option` 上。`Box<i32>` 拥有一个堆上的整数，解引用目标是 `i32`。

| 写法（`slot` 是 `Option<Box<i32>>`） | 得到的类型 | 做什么 | 原 `slot` |
|---|---|---|---|
| `&slot` | `&Option<Box<i32>>` | 借用整个 Option | 不变 |
| `slot.as_ref()` | `Option<&Box<i32>>` | 有盒子就借用盒子 | 不变 |
| `slot.as_deref()` | `Option<&i32>` | 有盒子就借用里面的整数 | 不变 |
| `slot.take()` | `Option<Box<i32>>` | 把盒子的所有权取出来 | 变成 `None` |

先比较两种借用：

```rust
let mut slot: Option<Box<i32>> = Some(Box::new(42));

let boxed: Option<&Box<i32>> = slot.as_ref();
let number: Option<&i32> = slot.as_deref();

println!("{boxed:?} / {number:?}");
println!("{slot:?}"); // 原值仍在，没有移动 Box，也没有复制整数
```

```text
slot：Some(Box ──拥有──> 42)
             ↑            ↑
as_ref() ────┘            │
得到 Some(&Box<i32>)      │
                          │
as_deref() ───────────────┘
得到 Some(&i32)
```

`Option` 这一层仍然保留：原来是 `Some`，返回包着引用的 `Some`；原来是 `None`，返回 `None`。
`as_ref()` 不等于 `&slot`：前者把引用放在 Option 里面，后者是引用整个 Option。

再接着取走盒子：

```rust
// 接上例；boxed、number 后面不再使用，前面的共享借用已结束
let extracted: Option<Box<i32>> = slot.take();

assert!(slot.is_none());
assert_eq!(extracted.as_deref(), Some(&42));
assert!(slot.take().is_none()); // 第二次已经没有盒子可取
```

```text
take 前：slot      = Some(Box ──拥有──> 42)

take 后：slot      = None
         extracted = Some(原来的 Box ──拥有──> 原来的 42)
```

这里移动的是原来的 Box，没有创建新盒子，没有复制堆上的整数。
`extracted.as_deref()` 只是借用整数做断言，仍然由 `extracted` 拥有 Box。
`slot` 本身没有被移动，所以还能使用，只是里面已经是 `None`。

| 方法 | 对接收者的要求 | 通用结果 |
|---|---|---|
| `as_ref()` | 共享借用 `&self` | `Option<T>` → `Option<&T>` |
| `as_deref()` | 共享借用 `&self`，且 `T` 实现 `Deref` | `Option<T>` → `Option<&T::Target>` |
| `take()` | 可变借用 `&mut self` | 返回原来的 `Option<T>`，给原位置留下 `None` |

因此 `slot` 需要声明为 `mut` 才能调用 `take()`；只调用前两者不需要。
如果后面还要使用 `boxed` 或 `number`，它们的共享借用仍然有效，就不能先调用 `take()`，
因为取走 Box 会让旧引用失去依托。

参见标准库 [Option::as_ref](https://doc.rust-lang.org/std/option/enum.Option.html#method.as_ref)、
[Option::as_deref](https://doc.rust-lang.org/std/option/enum.Option.html#method.as_deref) 与
[Option::take](https://doc.rust-lang.org/std/option/enum.Option.html#method.take)。

## `Result<T, E>` 是什么

可以把标准库中的 `Result` 简化理解为：

```rust
enum Result<T, E> {
    Ok(T),
    Err(E),
}
```

其中：

- `T` 是成功值的类型；
- `E` 是错误值的类型；
- `Ok(value)` 携带成功结果；
- `Err(error)` 携带失败原因。

调用方必须决定如何处理这两种情况：

```rust
match GatewayConfig::from_json(input) {
    Ok(config) => {
        // 使用成功解析的配置
    }
    Err(error) => {
        // 记录、转换或继续返回错误
    }
}
```

与异常不同，`Result` 是普通的枚举值，成功和失败都体现在函数签名与类型检查中。

## 自查与关联

as_ref、as_deref 和 take 分别留下什么？

[? 与 ??：成功继续，失败提前返回](question-mark.md) · [From 与 Into：为什么有两个转换入口](from-and-into.md) · [TryFrom、TryInto 与可能失败的转换](fallible-conversions.md)
