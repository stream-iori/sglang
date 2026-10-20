# 解构：从 State、结构体和元组中取值

[分类目录](index.md) · [总入口](../../index.md) · [问题索引](../../questions.md)

**模式只拆开已经收到的值；框架提取参数和 Rust 模式解构是两个步骤。**

| 阅读准备 | 内容 |
|---|---|
| 前置 | [struct、enum、impl 与 Self](../values/struct-enum-and-impl.md) |
| 源码入口 | [src/gateway.rs](../../../../src/gateway.rs)：`State(state): State<Arc<GatewayState>>` |
| 阅读时机 | 按当前课程需要阅读，不据此跳过 learning 的阶段验收。 |

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

框架怎样根据参数类型选择提取规则，见 [Extractor 原理：State、泛型与 trait 的最小例子](../../libraries/axum/extractor-mechanics.md)。

默认绑定方式是按值绑定。因为 handler 收到的是一个拥有所有权的 `State<AppState>`，所以
`State(state)` 会把内部 `AppState` 移动给局部变量 `state`，而不是借用它。

## State 是什么类型

```rust
struct State<T>(T);
```

它是一个泛型元组结构体。字段没有名字，通过 `.0` 访问。

| 写法 | 含义 |
|---|---|
| `State<T>` | 内部字段类型为 T 的包装类型 |
| `State<AppState>` | 内部字段类型为 AppState |
| `State(data)` | 表达式：创建包装值，把 data 放进去 |
| `wrapper.0` | 访问包装值的第一个字段 |
| `let State(data) = wrapper;` | 模式解构：拆开包装，把内部值绑定给 data |

```text
wrapper: State<AppState>
└── .0: AppState
    └── name: String
```

只定义这个结构体，还不会自动获得“提取”能力。规则来自后面的 trait 实现。

## 自查与关联

State 是结构体、表达式还是模式？

[match 分支与守卫的执行顺序](match-and-guards.md) · [if let 与 let else：成功值在哪里可用](if-let-and-let-else.md) · [ref、引用模式与解引用](reference-patterns.md)
