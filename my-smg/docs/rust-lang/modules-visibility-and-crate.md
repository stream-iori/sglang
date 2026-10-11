# Rust：`crate`、`mod`、`pub`、`use` 与模块路径

## 一句话

```text
crate：一份被编译的代码树
mod：给代码树分模块
pub：决定模块/项目外能否访问
use：把远处的名字引入当前作用域
```

## 先看全貌

```text
Cargo package
  │
  ├─ src/lib.rs   -> library crate 的根模块
  └─ src/main.rs  -> binary crate 的根模块

一个 package 可以有多个 crate；`crate::` 指“当前正在编译的那个 crate”。
```

假设 library crate 的目录是：

```text
src/
├─ lib.rs
├─ config.rs
└─ routers/
   ├─ mod.rs
   └─ http.rs
```

`lib.rs`：

```rust
pub mod config;
pub mod routers;
```

这会形成模块树：

```text
crate（根模块：lib.rs）
├─ config
└─ routers
   └─ http
```

## `crate::x`：从哪里开始找

```rust
use crate::config::Config;
```

意思不是“创建 crate”，而是：

```text
从当前 crate 根模块开始
  -> 找 config 模块
  -> 找 Config
```

这是编译期的模块路径解析，不是运行时查文件，也不创建对象。

```rust
// 在 crate::routers::http 内
crate::config::Config  // 从根找：crate -> config -> Config
self::helper           // 从当前模块找：crate::routers::http::helper
super::Router           // 从父模块找：crate::routers::Router
tokio::spawn(...)       // 从依赖 crate tokio 找
```

| 前缀 | 起点 |
|---|---|
| `crate::` | 当前 crate 根模块 |
| `self::` | 当前模块 |
| `super::` | 父模块 |
| `依赖名::` | 外部依赖 crate |

## `mod`：声明模块

```rust
mod config;      // 声明 config 模块，但模块对外不可见
pub mod config;  // 声明 config 模块，且允许 crate 外通过路径访问
```

`mod config;` 通常会对应 `config.rs` 或 `config/mod.rs`；它决定代码属于模块树的哪个节点。

```text
mod = 建模块树
```

## `pub`：开放访问权限

默认私有。`pub` 只改变可见性，不会自动把父模块也公开。

```rust
pub mod user {          // 模块对外公开
    pub struct User {   // 类型对外公开
        name: String,   // 字段仍然私有
    }

    impl User {
        pub fn new(name: String) -> Self {
            Self { name }
        }
    }
}
```

| 写法 | 可见范围 |
|---|---|
| 无 `pub` | 当前模块及其子模块 |
| `pub` | 对外公开（前提是沿途父模块也公开） |
| `pub(crate)` | 仅当前 crate 内公开 |
| `pub(super)` | 仅父模块公开 |

## `use`：引入名字，不引入代码

```rust
use crate::config::Config;

let config: Config;
```

没有 `use` 时也能写完整路径：

```rust
let config: crate::config::Config;
```

所以：

```text
mod = 声明/组织代码
use = 当前作用域的名字简写
```

`use` 不复制代码、不创建对象、不改变 `pub` 权限。

## `pub use`：重新导出

```rust
// lib.rs
pub mod user;
pub use user::User;
```

外部调用方本来要写：

```rust
use my_app::user::User;
```

重新导出后可以写：

```rust
use my_app::User;
```

```text
pub use = 允许外部从一个更短、更稳定的路径访问已有项目
```

## 最小记忆图

```text
crate
  └─ pub mod user
       └─ pub struct User

当前文件：use crate::user::User
外部项目：use my_app::user::User
```

```text
crate::：从项目当前 crate 的根开始找
mod：     放进模块树
pub：     开放访问
use：     当前文件少写路径
pub use： 对外提供更短路径
```
