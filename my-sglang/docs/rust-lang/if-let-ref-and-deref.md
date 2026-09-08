# Rust：`if let`、`ref` 与解引用

## 结论

下面这段代码的作用是：读取默认路由 ID；如果存在，就借用它查询路由表。

```rust
if let Some(ref default_id) = *default_router {
    self.routers.get(default_id).map(|r| r.clone())
} else {
    None
}
```

其中：

| 语法 | 作用 | 结果类型 |
|---|---|---|
| `*default_router` | 解引用读锁守卫，访问锁保护的值 | `Option<RouterId>` |
| `Some(...)` | 只处理存在默认路由的情况 | 匹配 `Option::Some` |
| `ref default_id` | 借用内部的路由 ID，不转移所有权 | `&RouterId` |

更直观的现代写法是：

```rust
if let Some(default_id) = default_router.as_ref() {
    self.routers.get(default_id).map(|r| r.clone())
} else {
    None
}
```

`as_ref()` 明确表达了“把 `Option<RouterId>` 转成 `Option<&RouterId>`”，通常比
`Some(ref default_id) = *default_router` 更容易读懂。

## `if let` 是什么

普通 `if` 判断 `bool`：

```rust
if is_ready { /* is_ready: bool */ }
```

`if let` 判断“值能否匹配模式”，并在匹配成功时声明模式中的变量：

```rust
let value = Some("hello");

if let Some(xx) = value {
    // xx: "hello"
}
```

```text
value = Some("hello")
       │
       ├─ 匹配 Some(xx) 成功 -> 声明 xx -> 执行 block
       └─ value 是 None      -> 跳过 block
```

它等价于只关心一个分支的 `match`：

```rust
match value {
    Some(xx) => { /* 使用 xx */ }
    _ => {}
}
```

所以 `Some(default_id)` 中：`Some` 是 `Option` 的变体，`default_id` 是在模式中声明的新变量。
`if let` 适合只处理一种情况；需要区分多个变体时使用 `match`。

## 实际类型链

[`RouterManager`](../../../sgl-model-gateway/src/routers/router_manager.rs) 中的字段类型是：

```rust
default_router: Arc<std::sync::RwLock<Option<RouterId>>>
```

读取后，各层类型如下：

```text
Arc<RwLock<Option<RouterId>>>
             │ .read()
             ▼
Result<RwLockReadGuard<Option<RouterId>>, PoisonError<_>>
             │ .unwrap_or_else(...)
             ▼
RwLockReadGuard<Option<RouterId>>       ← default_router
             │ *
             ▼
Option<RouterId>
             │ Some(ref default_id)
             ▼
&RouterId                               ← default_id
```

## 为什么使用 `*`

局部变量 `default_router` 不是 `Option<RouterId>`，而是：

```rust
RwLockReadGuard<'_, Option<RouterId>>
```

`RwLockReadGuard<T>` 实现了 `Deref<Target = T>`。因此：

```rust
*default_router
```

表示访问守卫后面的 `Option<RouterId>`。这里不会释放锁；只要守卫还在作用域内，读锁
就仍然有效。

模式匹配不会像方法调用那样自动穿透 `RwLockReadGuard`，所以原写法需要显式 `*`。

## 为什么使用 `ref`

如果写成：

```rust
if let Some(default_id) = *default_router {
```

`default_id` 会按值绑定，Rust 会尝试把 `RouterId` 从锁保护的数据中移出来：

```text
Option<RouterId>（锁中的共享数据）
       │ move RouterId
       ▼
default_id: RouterId
```

读锁只允许共享读取，不允许取走内部数据，因此编译器会拒绝这种移动。

加上 `ref` 后，模式绑定改为借用：

```text
Option<RouterId>（锁中的共享数据）
       │ borrow
       ▼
default_id: &RouterId
```

注意：`ref` 只改变模式变量的绑定方式，它不是变量类型的一部分。

```rust
Some(ref default_id) // default_id: &RouterId
Some(default_id)     // default_id: RouterId，默认按值绑定
```

## 推荐写法

### 写法一：使用 `as_ref()`

```rust
let default_router = self
    .default_router
    .read()
    .unwrap_or_else(|e| e.into_inner());

if let Some(default_id) = default_router.as_ref() {
    self.routers.get(default_id).map(|r| r.clone())
} else {
    None
}
```

类型变化很直接：

```text
Option<RouterId> --as_ref()--> Option<&RouterId>
```

方法调用会自动解引用 `RwLockReadGuard`，找到 `Option::as_ref()`。

### 写法二：返回值本来就是 `Option` 时使用 `?`

如果当前函数返回 `Option<Arc<dyn RouterTrait>>`，还可以写成：

```rust
let default_router = self
    .default_router
    .read()
    .unwrap_or_else(|e| e.into_inner());

let default_id = default_router.as_ref()?;
self.routers.get(default_id).map(|r| r.clone())
```

当默认路由为 `None` 时，`?` 会直接返回 `None`。

## 三种写法对照

| 写法 | `default_id` 类型 | 是否移动数据 | 建议 |
|---|---|---:|---|
| `Some(default_id) = *guard` | `RouterId` | 是，无法从读锁守卫中移动 | 不使用 |
| `Some(ref default_id) = *guard` | `&RouterId` | 否 | 正确，但不够直观 |
| `Some(default_id) = guard.as_ref()` | `&RouterId` | 否 | 推荐 |

## 一句话记忆

```text
*：穿过锁守卫，看到里面的 Option
ref：不要拿走 Some 里的值，只借来看
as_ref()：先把 Option<T> 变成 Option<&T>，意图最清楚
```
