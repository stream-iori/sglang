# Extractor 原理：用最小例子理解 State、泛型与 trait

[返回索引](index.md) · 前置：[Trait 与泛型](traits-dispatch-and-smart-pointers.md)
· 相关：[State 解构语法](if-let-ref-and-deref.md#statestate解构元组结构体)

## 先看结论

**不是从请求里“提取出类型”，而是参数类型决定提取规则，再由框架执行规则、生成参数值。**

```text
你提供的应用状态                  handler 声明的参数类型
AppState 值                      State<AppState>
      │                                 │
      └──────────────┬──────────────────┘
                     v
          框架调用对应的 trait 方法
                     │
                     v
        构造 State<AppState>，传给 handler
```

| 部分 | 谁定义 | 职责 |
|---|---|---|
| `AppState` | 应用开发者 | 描述业务数据 |
| `State<T>` | 框架 | 包装状态，同时作为选择提取规则的类型 |
| 提取器 trait | 框架 | 约定怎样准备参数 |
| trait 的实现 | 框架或扩展作者 | 给出具体提取步骤 |
| handler | 应用开发者 | 接收已经准备好的参数，处理业务 |

本文只保存独立教学示例，不增加项目实现、依赖或测试文件，也不代表要在 my-smg 中实现一套框架。
这里使用同步函数和自定义 `FromState`，有意去掉 HTTP、异步、错误响应等细节。

<a id="minimal-example"></a>

## 最小例子

```rust
// 1. 你定义的应用数据
#[derive(Clone)]
struct AppState {
    name: String,
}

// 2. 框架定义的包装类型
struct State<T>(T);

// 3. 框架约定：参数必须说明怎样从应用状态中取得自己
trait FromState<S> {
    fn from_state(source: &S) -> Self;
}

// 4. State<T> 的提取规则：克隆数据，再包装
impl<T: Clone> FromState<T> for State<T> {
    fn from_state(source: &T) -> Self {
        State(source.clone())
    }
}

// 5. 模拟框架：准备参数，然后调用你的函数
fn dispatch<S, E: FromState<S>>(
    source: &S,
    handler: fn(E),
) {
    let argument = E::from_state(source);
    handler(argument);
}

// 6. 你只声明自己需要 State<AppState>
fn health(wrapper: State<AppState>) {
    println!("{}", wrapper.0.name);
}

fn main() {
    let state = AppState {
        name: String::from("gateway"),
    };

    dispatch(&state, health);
}
```

按代码逻辑推演，最后会打印 `gateway`。本文没有执行这个示例。

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

<a id="from-state-contract"></a>

## trait 怎样描述提取规则

```rust
trait FromState<S> {
    fn from_state(source: &S) -> Self;
}
```

| 部分 | 大白话 |
|---|---|
| `S` | 数据来源的类型 |
| `source: &S` | 借用来源数据，不直接把它搬走 |
| `Self` | 实现这个 trait 的目标类型 |
| `-> Self` | 返回一个准备好的目标值 |

`from_state` 没有 `self` 参数，它是关联函数，不需要先创建一个目标对象才能调用。

接着为包装类型实现它：

```rust
impl<T: Clone> FromState<T> for State<T> {
    fn from_state(source: &T) -> Self {
        State(source.clone())
    }
}
```

对本例而言，T 是 AppState：

```text
来源类型：T = AppState
目标类型：Self = State<AppState>

&AppState → 克隆出 AppState → 包装成 State<AppState>
```

因为只拿到了来源的共享引用，不能把里面的 AppState 直接移动出去，所以本例使用 Clone 创建一个值。
这里克隆 String 会复制字符串数据；若状态字段是 Arc，克隆句柄则可以继续共享底层对象。
“提取”不等于“永远克隆”，这是本例选定的具体规则。

<a id="type-inference"></a>

## dispatch 如何知道该调用哪个实现

最关键的一次调用是：

```rust
dispatch(&state, health);
```

把它和函数签名对上：

```rust
fn dispatch<S, E: FromState<S>>(
    source: &S,
    handler: fn(E),
)
```

| 信息来源 | 编译器得到什么 |
|---|---|
| `state` 的类型是 AppState | `S = AppState` |
| `health` 接收 State<AppState> | `E = State<AppState>` |
| `E: FromState<S>` | 必须存在 `State<AppState>: FromState<AppState>` 的实现 |

health 是函数项，传给这里的 `fn(E)` 参数时会转换成对应的函数指针。本例不支持捕获环境的闭包。

类型确定后，下面这一行：

```rust
let argument = E::from_state(source);
```

在本次调用中对应：

```rust
let argument = State::<AppState>::from_state(source);
```

如果要明确指出 trait，可以写成完整限定形式：

```rust
let argument =
    <State<AppState> as FromState<AppState>>::from_state(source);
```

这三段是在解释同一次调用，不是需要重复执行三次的步骤。

```text
dispatch(&state, health)
    │
    ├── 根据 &state 确定 S
    ├── 根据 health 的参数确定 E
    ├── 检查 E 是否实现 FromState<S>
    │
    v
E::from_state(source)
    │
    v
State(source.clone())
    │
    v
health(argument)
```

**提取规则在编译期确定，具体数据在运行时取得。**
程序不需要扫描参数名、解析类型字符串或使用 Java 式运行时反射。
如果去掉这个 trait 实现，原来的 dispatch 调用无法通过编译；它不会等到请求到来后才猜该怎么办。

<a id="extract-vs-destructure"></a>

## 提取和解构是两步

| 步骤 | 对应代码 | 做什么 |
|---|---|---|
| 提取 | `E::from_state(source)` | 根据来源生成包装参数 |
| 调用 | `handler(argument)` | 把参数传给业务函数 |
| 访问字段 | `wrapper.0` | 从收到的包装里取出内部值 |

health 可以改写成先解构的形式：

```rust
fn health(wrapper: State<AppState>) {
    let State(state) = wrapper;
    println!("{}", state.name);
}
```

也可以在参数位置直接解构：

```rust
fn health(State(state): State<AppState>) {
    println!("{}", state.name);
}
```

这里最后一种写法更短，但初学时可以保留 `wrapper` 和 `.0`，分开阅读提取与取值。
具体语法见 [State(state)：解构元组结构体](if-let-ref-and-deref.md#statestate解构元组结构体)。

<a id="axum-mapping"></a>

## 和真实 Axum 怎么对应

| 教学例子 | 真实 Axum |
|---|---|
| `AppState` | 你定义的业务状态 |
| `struct State<T>(T)` | Axum 定义的 `pub struct State<S>(pub S)` |
| main 中的 state | 通过 `Router::with_state` 提供的状态 |
| `FromState` | 本例自定义；Axum 使用 `FromRequestParts` 等 trait |
| `State<T>` 的克隆实现 | Axum 通过 `FromRef` 取得所需状态；同类型的常见实现使用 Clone |
| `dispatch` | 模拟 Handler 调用链中准备参数的部分 |
| 同步 health | 实际通常是异步 handler |

Axum 的 State 提取器还能从较大的状态中取得子状态，并受相应线程安全约束。
完整 HTTP 流程还涉及路由、请求、提取失败、Future 和响应转换，本文没有实现这些机制。

```text
State<AppState>  → 从应用状态准备 AppState
Json<AppState>   → 从请求体 JSON 解析 AppState
```

上面两种写法中，AppState 表示目标数据类型，外面的 State 或 Json 表示不同的提取规则。
本例只实现 State 这一种；同一个 dispatch 也只能使用已有 FromState 实现的目标类型。

## 自查理解

| 问题 | 应能说明的重点 |
|---|---|
| State 是 Rust 关键字吗？ | 不是，本例是自定义结构体，真实 State 由 Axum 定义 |
| 把 wrapper 改名会影响提取吗？ | 不会，类型和 trait 实现决定规则 |
| 为什么只定义 State 还不够？ | 包装结构和提取规则是两件事 |
| 谁调用 health？ | dispatch；真实场景中由框架调用链完成 |
| 为什么需要 Clone？ | 本例从共享引用构造一个拥有所有权的参数值 |
| 解构是否负责寻找应用状态？ | 不负责，它只拆开已经拿到的值 |

## 关联阅读与源码

- [模式解构、if let、let ... else、ref 与解引用](if-let-ref-and-deref.md)
- [Trait、静态分发、动态分发与智能指针](traits-dispatch-and-smart-pointers.md)
- [Axum：post、Handler、闭包与 Future](axum-routing-handler-and-future.md)
- [Axum 0.8.9：提取器说明](https://docs.rs/axum/0.8.9/axum/extract/index.html)
- [Axum 0.8.9：State 定义及提取实现](https://github.com/tokio-rs/axum/blob/axum-v0.8.9/axum/src/extract/state.rs)
