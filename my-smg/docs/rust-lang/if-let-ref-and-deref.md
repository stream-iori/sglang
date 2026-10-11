# Rust：模式解构、`if let`、`let ... else`、`ref` 与解引用

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

## `let ... else`：取出值，失败就提前离开

`my-smg` 的节点选择返回 `Option<usize>`：`Some(index)` 表示选中了节点，`None` 表示没有
可用节点。`chat` 可以这样处理：

```rust
let Some(index) = selected_index else {
    return (
        StatusCode::SERVICE_UNAVAILABLE,
        "no healthy worker".to_string(),
    );
};

// 从这里开始可以使用 index: usize
```

执行路径：

```text
selected_index: Option<usize>
        │
        ├─ Some(1) → 绑定 index = 1 → 继续执行
        └─ None    → 执行 else 中的 return → 结束整个 chat 函数
```

它相当于：

```rust
let index = match selected_index {
    Some(value) => value,
    None => {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "no healthy worker".to_string(),
        );
    }
};
```

这里的 `return` 返回的是**整个函数**的结果，不只是离开 `else`。`else` 分支必须明确离开
当前控制流，例如 `return`；不能在 `else` 中给 `index` 随意提供一个默认值后继续。

| 写法 | 匹配成功后，变量能用到哪里？ | 匹配失败时 |
|---|---|---|
| `if let Some(index) = value { ... }` | 只在 `{ ... }` 内 | 跳过该块，或进入 `else` 块 |
| `let Some(index) = value else { return ...; };` | 当前作用域后续代码 | 必须提前离开当前控制流 |

遇到 `Result<T, E>` 的失败传播时，还可以对照 [`?` 与提前返回](result-question-mark-and-error-propagation.md)；
`let ... else` 在这里用于把 `None` 映射成明确的 HTTP `503` 响应。

## `State(state)`：解构元组结构体

Axum 的 `State` 可以简化理解为一个只有一个字段的元组结构体：

```rust
pub struct State<S>(pub S);
```

同样的外形在不同位置有不同含义：

```rust
State(app_state) // 表达式：创建一个 State 值
State(state)     // 模式：拆开 State，把内部值绑定到 state
```

在 Axum handler 中：

```rust
async fn health(State(state): State<AppState>) {
    // state: AppState
}
```

冒号左右分别是：

```text
State(state) : State<AppState>
     │                │
     │                └─ 参数的类型
     └─ 参数使用的解构模式
```

它相当于先接收包装值，再手动拆开：

```rust
async fn health(wrapper: State<AppState>) {
    let State(state) = wrapper;
}
```

完整数据流是：

```text
Router 中的 AppState
        │ Axum State extractor
        v
State<AppState>
        │ State(state) 解构
        v
state: AppState
```

这里的“extractor”和“解构”不是一回事：

| 层次 | 负责什么 |
|---|---|
| Axum `State` extractor | 从 Router 的应用状态构造 `State<AppState>` 参数 |
| Rust `State(state)` 模式 | 从这个元组结构体中取出内部值 |

默认绑定方式是按值绑定。因为 handler 收到的是一个拥有所有权的 `State<AppState>`，所以
`State(state)` 会把内部 `AppState` 移动给局部变量 `state`，而不是借用它。

### 为什么 `AppState` 需要 `Clone`

`Router::with_state(state)` 会把 `state` 移动进 Router。之后 `main` 不能继续使用原变量，除非
传入的是 `state.clone()`：

```text
state ──move──> Router
```

`AppState: Clone` 的直接用途是让 Axum 在构建服务、分发请求时复制状态值或状态句柄。它不表示
每个 handler 都有一份完全独立的底层资源；如果状态内部以后放入 `Arc<T>`，克隆
`AppState` 只会克隆 `Arc`，多个请求仍可指向同一个 `T`。

响应结构体不需要 `Clone`，原因不是它“只活在 handler 内”：handler 会创建响应值，再把所有权
移动给 `Json` 和 HTTP 响应管线。这个过程只有一个所有者，没有复制需求。

```text
HealthResponse ──move──> Json<HealthResponse> ──move──> HTTP 响应管线
```

## 字段借用与 `match` 表达式

网关根据配置字段构造策略：

```rust
let policy: Box<dyn Policy + Send> = match &config.policy {
    PolicyKind::FirstHealthy => Box::new(FirstHealthy::new()),
    PolicyKind::RoundRobin => Box::new(RoundRobin::new()),
};
```

`.` 字段访问比前置的 `&` 结合得更紧，所以：

```rust
&config.policy
```

等价于：

```rust
&(config.policy) // 类型为 &PolicyKind：借用 policy 字段
```

当前 `PolicyKind` 没有实现 `Copy`。如果写 `match config.policy`，会从 `config` 中移动该字段；
后面就不能再把完整 `config` 移进 `GatewayState`。匹配 `&config.policy` 只读取字段，
不复制、不取走它。虽然匹配对象是 `&PolicyKind`，Rust 的模式匹配借用规则允许直接写
`PolicyKind::RoundRobin`，无需在每个分支再写 `&`。

另一个容易忽略的点：`match` 本身可以产生值。下面的分号只结束 `let` 语句，匹配结果仍被
赋给 `policy`：

```rust
let policy: Box<dyn Policy + Send> = match &config.policy {
    PolicyKind::FirstHealthy => Box::new(FirstHealthy::new()),
    PolicyKind::RoundRobin => Box::new(RoundRobin::new()),
};
```

但如果 `match` 是函数末尾的独立表达式，紧跟 `}` 的分号会丢弃其结果：

```rust
fn choose(flag: bool) -> i32 {
    match flag {
        true => 1,
        false => 2,
    } // 没有分号：整个 match 的值作为函数返回值
}
```

在 `my-smg` 的 `chat` 中，如果选中 worker 的 `match` 后加 `;`，又在后面写一个旧响应，
函数就会返回旧响应。编译可能通过，但新写的选择结果没有成为 HTTP 响应。

策略为什么需要在多个请求之间共享，见
[配置驱动的共享策略](traits-dispatch-and-smart-pointers.md#my-smg配置驱动的共享策略)。

<a id="match-guards"></a>

## match 分支守卫：模式后面还可以加条件

`my-smg` 的测试服务器清理函数有这一段：

```rust
match self.task.cancel_and_join().await {
    Ok(()) => Ok(()),
    Err(error) if error.is_cancelled() => Ok(()),
    Err(error) => Err(error),
}
```

`Err(error) if error.is_cancelled()` 分两步：先匹配 Err 并绑定 error，再检查布尔条件。
条件不满足时，继续尝试后面的分支；不是直接退出 match。

| 输入 | 选中的分支 | 清理结果 |
|---|---|---|
| 正常完成 | `Ok(())` | 成功 |
| 因取消结束 | 带 if 的 Err 分支 | 成功 |
| panic 等其他任务错误 | 最后一个 Err 分支 | 保留错误 |

普通模式负责匹配结构，分支守卫负责附加条件。守卫可以为 false，因此不能只靠一个带守卫的
分支覆盖所有 Err；这里保留无条件的最后一个分支。
参见 [Rust Reference：match guards](https://doc.rust-lang.org/reference/expressions/match-expr.html#match-guards)。

先用整数看分支顺序：

```rust
let value = Some(-2);
let label = match value {
    Some(number) if number > 0 => "positive",
    Some(_) => "non-positive",
    None => "missing",
};
assert_eq!(label, "non-positive");
```

```text
Some(-2)
   ↓ 第一个模式匹配成功：number = -2
number > 0 为 false
   ↓ 继续尝试第二个分支
Some(_) 匹配成功 → 返回 non-positive
```

这段逻辑适用于“服务器清理允许已完成”的场景。取消行为测试仍需要明确检查
`JoinError::is_cancelled()`，不能用“清理成功”替代“确实被取消”的断言。见
[Tokio 资源管理](tokio-basics-and-task-cancellation.md#managed-task-resource-lifecycle)。

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

锁守卫如何通过 `Deref/DerefMut` 访问内部数据、如何用 `Drop` 自动解锁，以及毒锁的含义，见
[`Mutex`、`MutexGuard`、解引用与毒锁](mutex-guard-deref-and-poisoning.md)。

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
let Some(index) = value else { return ...; }：有值就取出，没值就提前返回。
State(state)：按结构拆包装值，state 是新绑定的变量
*：穿过锁守卫，看到里面的 Option
ref：不要拿走 Some 里的值，只借来看
as_ref()：先把 Option<T> 变成 Option<&T>，意图最清楚
```
