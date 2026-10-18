# Rust：以 Java 9 模块系统理解 `crate`、`mod`、`pub` 与 `use`

## 先建立对应关系

**Java 9 的 JPMS `module` 最适合帮助理解 Rust 的 `crate` 边界；Java `package` 更接近 Rust `mod`。** 名字相似不代表层级相同。

```text
Java 9（JPMS）                     Rust
module                            crate
├── package api                   ├── mod api
│   └── public class User         │   └── pub struct User
└── package internal              └── mod internal
    └── 实现细节                       └── 实现细节
```

这是职责上的类比，不是语法或规则上的等价。本文 Java 对比以显式命名的 JPMS 模块为主，讨论普通代码访问；不展开自动模块、未命名模块和反射例外。

| 职责 | Java 9 | Rust | 不能直接等同的地方 |
|---|---|---|---|
| 工程与构建管理 | Maven/Gradle 工程模块 | Cargo package | 构建工具里的 module 不一定是 JPMS module |
| 依赖与封装边界 | JPMS `module` | `crate` | crate 还是 Rust 的独立编译单元 |
| 内部组织代码 | `package` | `mod` | Rust 模块树参与可见性判断 |
| 声明依赖 | `requires`，构建工具另负责取得依赖 | 通常在 `Cargo.toml` 中声明依赖 | Rust `use` 不负责取得依赖 |
| 对外开放 API | `exports` 包，加上类型和成员的访问权限 | `pub` 加可访问路径，或 `pub use` | Java 按包导出，Rust 可以重新导出具体项 |
| 引入名字简写 | `import` | `use` | 都不能绕过访问权限 |

Java 的模块与包规则见 [Java 9 语言规范第 7 章](https://docs.oracle.com/javase/specs/jls/se9/html/jls-7.html)。

## `crate`：像 JPMS module 一样理解边界

Java 的一个 JPMS module 可以包含多个 package；Rust 的一个 crate 可以包含多个 mod。两者都让你区分“内部实现”和“对外接口”。

但 Cargo package 和 crate 还要分开：**package 是 Cargo 管理构建、版本和发布的单位；crate 是 Rust 编译的单位。**

| 概念 | 管什么 | Java 对照 |
|---|---|---|
| Cargo package | 名字、版本、依赖和构建目标，由 `Cargo.toml` 描述 | 接近带 `pom.xml` 的 Maven 工程模块 |
| crate | 从一个根模块出发，独立编译一棵模块树 | 封装边界接近 JPMS module |
| `mod` | crate 内部组织类型、函数等项 | 接近 Java package |

```text
一个 Cargo package
├── src/lib.rs  → library crate 的根模块
└── src/main.rs → binary crate 的根模块
```

一个 package 可以包含至多一个 library crate，以及多个 binary crate。`lib.rs` 与 `main.rs` 即使在同一工程里，也属于两个 crate。[Rust：Packages and Crates](https://doc.rust-lang.org/book/ch07-01-packages-and-crates.html)

| 路径所在位置 | `crate::` 从哪里开始找 |
|---|---|
| `lib.rs` 及其子模块 | library crate 的根模块 |
| `main.rs` 及其子模块 | binary crate 的根模块 |

**`crate::` 指当前正在编译的 crate，不是整个 Cargo package。** 因此，库中的 `pub(crate)` 项不能仅因为 `main.rs` 在同一个 package 中，就被 binary crate 访问。

## `mod`：接近 Java package，但父子关系有实际作用

假设目录如下：

```text
src/
├── lib.rs
├── config.rs
└── routers/
    ├── mod.rs
    └── http.rs
```

只有声明了对应模块，才形成这棵树：根模块声明 `config` 和 `routers`，`routers` 再声明 `http`。

```text
crate（根模块：lib.rs）
├── config
└── routers
    └── http
```

| Rust 写法 | 意思 |
|---|---|
| `mod config;` | 声明 config 模块，默认私有 |
| `pub mod config;` | 声明 config 模块，并将它标记为公开；外部访问仍须有可访问路径 |

文件形式的模块通常对应 `config.rs` 或 `config/mod.rs`，也可以把模块内容写在声明处。**文件存在不代表自动加入模块树，`use` 也不能代替模块声明。** [Rust：模块组织](https://doc.rust-lang.org/book/ch07-02-defining-modules-to-control-scope-and-privacy.html)

<a id="module-declaration-to-file"></a>

### 从 `pub mod gate;` 到 `circuit_breaker/gate.rs`：逐层找文件

**不是文件夹自动关联，而是 `mod` 声明让编译器按约定寻找模块内容。**
下面采用默认文件布局，不使用 `#[path]` 改写位置。

| 声明写在哪里 | 声明内容 | 编译器寻找的文件 |
|---|---|---|
| `src/lib.rs` | `pub mod circuit_breaker;` | `src/circuit_breaker.rs` |
| `src/circuit_breaker.rs` | `pub mod gate;` | `src/circuit_breaker/gate.rs` |

```text
src/lib.rs
  │ pub mod circuit_breaker;
  ▼
src/circuit_breaker.rs
  │ pub mod gate;
  ▼
src/circuit_breaker/gate.rs
  └─ 定义 CircuitGate 等类型
```

这里 `circuit_breaker.rs` 与同名目录 `circuit_breaker/` 可以同时存在：

```text
src/
├── lib.rs
├── circuit_breaker.rs       ← 父模块自己的内容，以及子模块声明
└── circuit_breaker/
    └── gate.rs              ← 子模块的内容
```

对应的模块路径是 `my_smg::circuit_breaker::gate::CircuitGate`。
文件层次用于定位源码，模块路径用于编译期解析名字，不是运行时加载文件。

#### 把 mod、pub、use 分开看

| 写法 | 负责什么 | 不负责什么 |
|---|---|---|
| `mod gate;` | 声明子模块，从文件加载定义 | 不自动向外公开模块 |
| `pub mod gate;` | 声明子模块，并开放模块可见性 | 不自动公开内部所有类型和字段 |
| `use gate::CircuitGate;` | 引入已有名字，后面可以写短名 | 不代替 mod 声明，也不绕过可见性 |

所以，将 `pub mod gate;` 改成 `mod gate;`，**寻找的文件不变，只是访问权限变了**。
外部能使用完整路径，还需要库根公开 circuit_breaker、父模块公开 gate，以及 CircuitGate 本身公开。

在 `gate.rs` 中写：

```rust
use super::{CircuitBreaker, CircuitState};
```

`super` 指模块树上的父模块 circuit_breaker，因此能找到父模块定义的类型。
不是“扫描上一级文件夹”，也不是 Java 的父类。

#### 与 mod.rs 写法的关系

| 模块 | 当前文件写法 | 另一种合法写法 |
|---|---|---|
| circuit_breaker | `src/circuit_breaker.rs` | `src/circuit_breaker/mod.rs` |
| circuit_breaker::gate | `src/circuit_breaker/gate.rs` | `src/circuit_breaker/gate/mod.rs` |

同一个模块只能选其中一种，不能同时用两份候选文件定义它，否则会产生歧义。
两种布局不改变模块路径，也不改变访问权限。

本项目 `gateway.rs` 中的 `mod retry;`、`#[cfg(test)] mod tests;` 遵循同样规则，
分别定位 `gateway/retry.rs` 和 `gateway/tests.rs`。tests 这个名字本身没有特殊加载规则，
`#[cfg(test)]` 才决定该模块只在测试构建中加入。

一句话：**`mod` 连上模块树，约定决定文件位置，`pub` 决定权限，`use` 简写名字。**
另见 [Cargo 如何识别 library 和 binary](workspace.md#cargo-如何识别-library-和-binary)：crate 根文件的发现是另一层规则。

Java 包名与 Rust 模块树最大的差别是：

| 情况 | Java 9 | Rust |
|---|---|---|
| 子包/子模块访问父级私有内容 | 子包没有特殊权限 | 子模块可以访问父模块的私有项 |
| 名称层级 | `com.example` 与 `com.example.config` 是不同的包 | `config` 可以是根模块的实际子模块 |
| 访问控制 | 包名层级不赋予额外权限 | 模块父子关系参与权限判断 |

Java 的包名可以有层级，但子包不会继承父包的包访问权限。[Java 9：Package Members](https://docs.oracle.com/javase/specs/jls/se9/html/jls-7.html#jls-7.1)

## 模块路径：从哪里开始找

以 `crate::config::Config` 为例：

```text
当前 crate 根模块 → config 模块 → Config 类型
```

这是编译期名字解析，不是运行时读取文件，也不会创建对象。

| Rust 路径 | 起点或含义 | Java 对照 |
|---|---|---|
| `crate::config::Config` | 从当前 crate 根模块找 | 接近使用完整限定名，但 Java 类名通常不带 JPMS module 名 |
| `self::helper` | 从当前模块找 | 不是 Java 的 `this`，不表示当前对象 |
| `super::Router` | 从父模块找 | 不是 Java 的 `super`，不表示父类 |
| `依赖名::类型名` | 从外部依赖 crate 找 | 接近访问依赖模块所提供的类型 |

**Rust 的模块路径表达组织关系，不表达类的继承关系。**

## `pub`：对照 Java 的 `public` 与 `exports` 两层检查

Java 9 中，一个顶层类型声明为 `public`，不代表其他模块一定能使用它：调用方还需要读取目标模块，目标模块也需要向调用方导出该包。[Java 9：模块声明](https://docs.oracle.com/javase/specs/jls/se9/html/jls-7.html#jls-7.7)

```text
Java：调用方读取目标 module
        → 目标 module 向调用方 exports 对应 package
          → 类型是 public
            → 所访问的成员也可见

Rust：调用方依赖目标 crate
        → 调用方能访问类型的公开路径
          → 类型是 pub
            → 所访问的成员也可见
```

| 场景 | Java 9 | Rust |
|---|---|---|
| 类型公开，但外部路径未开放 | 未导出包中的 `public` 类型，普通外部代码不能直接访问 | 私有模块中的 `pub` 类型，外部不能沿该私有模块路径访问 |
| 类型公开，成员私有 | `public` 类不使字段自动成为 `public` | `pub struct` 不使字段自动成为 `pub` |
| 内部跨包/模块使用 | 未导出包中的 `public` 类型可供同一 JPMS module 内跨包使用 | `pub(crate)` 项可供当前 crate 内使用，仍须满足路径可见性 |

Java 通过 `exports` 控制包边界，Rust 通过模块路径和项的可见性控制边界。Rust 没有额外的 `module-info.java` 来列出所有导出包。

### Rust 可见性表

| 写法 | 可见范围 | Java 对照提醒 |
|---|---|---|
| 无 `pub` | 定义所在模块及其子模块 | 不等同于 Java 默认的包访问权限 |
| `pub(super)` | 父模块及其子模块范围 | 不等同于 Java `protected`，没有继承语义 |
| `pub(in 路径)` | 指定祖先模块及其子模块范围 | 可以精确限定内部协作区域，Java 没有直接对应的修饰符 |
| `pub(crate)` | 当前 crate 内 | 接近模块内部 API 的意图，但 Java 没有对应修饰符 |
| `pub` | 允许公开访问，实际访问还要检查路径 | 类似 `public` 仍要配合边界开放 |

指定可见范围不保证范围内的所有位置都能沿任意路径访问；沿途模块也必须可访问。常见私有项遵循模块范围规则，公开枚举的变体等另有默认规则。[Rust：Visibility and Privacy](https://doc.rust-lang.org/reference/visibility-and-privacy.html)

**Rust 结构体的私有字段是模块范围的私有。** 同一模块里的普通函数也能访问它，不必是该结构体的方法。不能直接套用 Java “只有类内部才能访问 private 字段”的心智模型。

### 多一层模块时：`pub(super)` 与 `pub(in 路径)`

**`super` 只指直接父模块，不会自动开放给祖父模块。** 假设 `reset` 定义在 `internal::xxx::routing` 中：

```text
crate
└── internal                 ✗ 不能直接访问 reset
    └── xxx                  ✓ 能访问 reset
        ├── diagnostics      ✓ 能访问 reset
        └── routing          ✓ 能访问 reset
            └── reset        ← pub(super)，父模块是 xxx
```

`pub(super)` 的范围是整个 `xxx` 子树，不只是 `xxx` 模块本身。`internal` 位于该子树外，即使写完整路径 `xxx::routing::reset`，也不能绕过权限。

如果需要让 `internal` 也能访问，**不必改成完全公开的 `pub`，也不必调整目录结构**，可以指定祖先模块作为边界：

| `reset` 的可见性 | 范围 | `internal` 能直接访问吗 |
|---|---|---|
| 无 `pub` | `routing` 子树 | ✗ |
| `pub(super)` | `xxx` 子树 | ✗ |
| `pub(in crate::internal)` | `internal` 子树 | ✓ |
| `pub(crate)` | 当前 crate | ✓ |
| `pub` | 允许公开访问，实际范围仍受路径限制 | ✓ |

表中假设沿途模块对访问方可见。若只想在 `internal` 内部协作，`pub(in crate::internal)` 比 `pub(crate)` 或 `pub` 更精确：

```text
crate
├── api                      ✗ 在指定范围外
└── internal                 ✓ 指定边界
    ├── monitor              ✓ 在指定范围内
    └── xxx                  ✓ 在指定范围内
        └── routing          ✓ 在指定范围内
            └── reset        ← pub(in crate::internal)
```

还需要检查三个条件：

| 条件 | 本例怎么理解 |
|---|---|
| 路径必须指向定义模块的祖先 | `internal` 是 `routing` 的祖先，可以指定；兄弟模块 `api` 不可以 |
| 沿途模块也必须可见 | 若 `routing` 是 `xxx` 内的私有模块，`internal` 仍不能穿过它访问 `reset`；需要一并开放该模块路径 |
| 看模块树，而非仅看文件夹 | 单纯新建 `xxx` 文件夹不会改变 `super`；模块声明建立父子关系后才会改变 |

在 Rust 2018 及以后，`pub(in 路径)` 的路径须以 `crate`、`self` 或 `super` 开始，并直接解析到祖先模块，不能使用 `use` 引入的别名。[Rust：受限可见性](https://doc.rust-lang.org/reference/visibility-and-privacy.html#pubin-path-pubcrate-pubsuper-and-pubself)

## `use`：最接近 Java import

| 操作 | Java 9 | Rust |
|---|---|---|
| 写完整路径 | `com.example.config.Config` | `crate::config::Config` |
| 引入名字 | `import com.example.config.Config;` | `use crate::config::Config;` |
| 引入后引用类型 | `Config` | `Config` |

`import` 与 `use` 都让当前作用域少写路径。它们不创建对象、不复制代码，也不改变目标的访问权限。

```text
Java requires → 声明模块依赖/可读性
Java import   → 引入名字简写

Rust mod      → 声明模块
Rust use      → 引入名字简写
```

Java 的依赖声明与名字引入规则见 [Java 9 语言规范](https://docs.oracle.com/javase/specs/jls/se9/html/jls-7.html)；Rust 导入路径同样需要通过 [可见性检查](https://doc.rust-lang.org/reference/visibility-and-privacy.html)。

## `pub use`：提供公开入口，Java exports 不会改路径

假设 `User` 在内部 `user` 模块中定义，并声明为 `pub`。在 crate 根模块用 `pub use` 重新导出它，可以提供另一条路径：

```text
内部组织路径：my_app::user::User
                    │
                 pub use
                    ↓
对外入口路径：my_app::User
```

| 操作 | 效果 |
|---|---|
| Java `exports com.example.user` | 向外开放该包，类型仍叫 `com.example.user.User` |
| Rust `pub use` 重新导出 `User` | 建立另一个访问路径，例如 `my_app::User` |
| Java 普通 `import` | 只简化当前文件中的名字，不建立对外入口 |

**内部 `user` 模块可以保持私有，同时通过根模块公开重新导出其中的公开 `User`。** 外部只能使用开放的入口，不能因此访问内部模块的其他内容。重新导出不会创建新类型或包装对象。[Rust：Re-exporting and Visibility](https://doc.rust-lang.org/reference/visibility-and-privacy.html#re-exporting-and-visibility)

所以，“沿途父模块也要公开”是外部沿原模块路径访问时的要求；`pub use` 可以提供另一条公开路径。

### 文件布局、模块路径与公开入口可以分别设计

**Rust 更明显的优势是：内部文件布局、模块组织、对外 API 路径可以分别设计。** 假设 `Router` 是公开类型，并在 crate 根通过 `pub use` 重新导出：

```text
物理文件：src/internal/xxx/routing.rs
模块路径：crate::internal::xxx::routing::Router
公开入口：gateway_demo::Router
                           ↑
                        pub use
```

这里对外路径变短了，内部目录并没有变浅。`pub use` 隐藏了调用者不需要了解的组织层级；内部仍按职责划分模块，对外可以提供简短、稳定的入口。

| 层面 | 决定什么 | 本例 |
|---|---|---|
| 物理文件布局 | 源码存放在哪里 | `src/internal/xxx/routing.rs` |
| 模块组织 | 类型在模块树中的位置与可见性边界 | `crate::internal::xxx::routing::Router` |
| 对外 API 路径 | 调用者通过什么入口使用类型 | `gateway_demo::Router` |

默认的文件模块布局仍映射模块路径；内联模块和 `#[path]` 可以让模块组织与文件位置有所不同。`pub use` 改变的是公开入口，不会移动文件，也不会改变原模块的父子关系。[Rust：模块与文件位置](https://doc.rust-lang.org/reference/items/modules.html)、[重新导出](https://doc.rust-lang.org/reference/visibility-and-privacy.html#re-exporting-and-visibility)

Java 常用的 `com.company.product` 前缀用于避免包名冲突；Rust 的外部路径从 crate 名开始，通常不需要同样的反向域名前缀。这也会让 Rust 工程看起来更浅。Java 的包命名约定见 [Java 9 语言规范](https://docs.oracle.com/javase/specs/jls/se9/html/jls-7.html#jls-7.4.1)。

**路径较短不代表目录必须较浅。** Rust 没有“三层以内”的语言限制；目录深度应由实际职责决定。

### Java 9 另外三个关键词不要混淆

| Java 9 写法 | 含义 | 与 Rust 的区别 |
|---|---|---|
| `requires transitive` | 让读取当前模块的其他模块也能读取指定依赖模块 | 不是 `pub use`，不会给类型建立新路径 |
| `exports ... to ...` | 只向指定模块导出包 | `pub(crate)`、`pub(super)` 限制的是内部范围，不能指定某个外部 crate |
| `opens` | 控制包对反射的开放，支持深度反射访问 | 不是普通代码的导出权限；Rust 的 `pub` 没有对应的反射开放语义 |

上述机制见 [Java 9：依赖](https://docs.oracle.com/javase/specs/jls/se9/html/jls-7.html#jls-7.7.1)与[导出和开放包](https://docs.oracle.com/javase/specs/jls/se9/html/jls-7.html#jls-7.7.2)。

## 最小记忆图

```text
边界：      Java JPMS module  ≈ Rust crate
内部组织：  Java package      ≈ Rust mod
名字简写：  Java import       ≈ Rust use
对外 API：  Java exports + public
            ≈ Rust 公开路径 + pub，或 pub use
```

| Rust 关键词 | 一句话记忆 |
|---|---|
| `crate::` | 从当前 crate 根模块找 |
| `mod` | 声明模块，建立模块树 |
| `pub` | 开放项的访问权限 |
| `pub(crate)` | 把权限范围限制在当前 crate 内 |
| `pub(in 路径)` | 把权限范围限制在指定祖先模块的子树内 |
| `use` | 在当前作用域少写路径 |
| `pub use` | 为已有项提供公开访问入口 |

## 一个完整例子：把所有概念串起来

下面是一个假想的 `gateway-demo` 工程，只用于理解组织与访问规则，不代表当前项目已有这些文件。假设使用 Cargo 默认的目标发现规则，且各文件按下文说明声明模块。

### 1. 先看文件，再数 package 和 crate

```text
gateway-demo/                      ← 1 个 Cargo package
├── Cargo.toml                     ← 名字、版本、依赖、构建目标
└── src/
    ├── lib.rs                     ← library crate 的根
    ├── api.rs                     ← 库的 api 模块
    ├── internal/
    │   ├── mod.rs                 ← 库的 internal 模块
    │   └── routing.rs             ← internal 的子模块
    ├── main.rs                    ← 第一个 binary crate 的根
    └── bin/
        └── inspect.rs             ← 第二个 binary crate 的根
```

| 数量 | 对象 | 职责 |
|---|---|---|
| 1 个 package | `gateway-demo` | 一起管理版本、依赖和发布 |
| 1 个 library crate | 根是 `lib.rs` | 提供公开 API 和内部实现 |
| 2 个 binary crate | 根分别是 `main.rs`、`inspect.rs` | 分别提供启动程序和检查程序 |

所以这里共 **3 个 crate**。`api.rs`、`routing.rs` 是库的模块文件，不因为独立成文件就成为新 crate。假设采用默认库名，Rust 中用 `gateway_demo` 引用这个库：package 名中的连字符在默认库名中变成下划线。

### 2. 文件如何进入模块树：`mod` 与 `pub mod`

假设库根声明公开的 `api` 模块和私有的 `internal` 模块；`internal` 再声明仅 crate 内可见的 `routing` 子模块。两个 binary 不重复声明这些库模块，而是使用库的公开入口。

```text
library crate：gateway_demo（lib.rs）
├── api                            ← pub mod：公开路径
│   └── Gateway                    ← pub 类型
└── internal                       ← mod：对外隐藏
    ├── normalize                  ← 私有函数
    └── routing                    ← pub(crate) mod：仅库内部
        ├── choose                 ← pub(crate) 函数
        ├── reset                  ← pub(super) 函数
        └── cursor                 ← 私有函数

binary crate：启动程序（main.rs）
└── main

binary crate：检查程序（inspect.rs）
└── main
```

这三棵树分别编译。`mod` 负责声明树上的节点；`pub mod` 还开放模块的可见性。`use` 只引入已有名字，不建立这些模块。

### 3. 用一次访问检查理解全部可见性

再假设 `Gateway` 有公开方法 `status` 和私有字段 `name`；各访问路径没有其他权限障碍。

| 项 | 谁能访问 | 谁不能访问 | 原因 |
|---|---|---|---|
| `Gateway` | 两个 binary 和库内部 | — | 类型公开，且 `api` 路径公开 |
| `Gateway` 的 `status` 方法 | 两个 binary 和库内部 | — | 类型、路径、方法都可见 |
| `Gateway` 的 `name` 字段 | `api` 模块及其子模块 | 库的兄弟模块、两个 binary | 字段私有，边界是定义模块 |
| `normalize` | `internal` 及其子模块，包括 `routing` | `api`、库根、两个 binary | 无 `pub`，仅定义模块及其后代可见 |
| `choose` | 库根、`api`、`internal`、`routing` | 两个 binary | `pub(crate)` 只覆盖 library crate |
| `reset` | `internal` 及其子模块 | `api`、库根、两个 binary | `pub(super)` 将范围扩大到父模块 `internal` |
| `cursor` | `routing` 及其子模块 | `internal`、`api`、库根、两个 binary | 子模块私有项不会反向开放给父模块 |

为什么 `api` 能通过路径访问私有的 `internal` 模块？因为 `internal` 声明在库根，库根及其所有后代都能访问这个模块；进入模块后，里面的函数仍各自检查权限。**模块可见，不代表它的全部内容都可见。**

### 4. 同一个名字，从不同位置怎么找：路径与 `use`

| 使用位置 | 路径或名字 | 解释 |
|---|---|---|
| 库的 `internal::routing` 中 | `crate::api::Gateway` | 从 library crate 根找公开类型 |
| 库的 `internal::routing` 中 | `self::cursor` | 从当前 `routing` 模块找私有函数 |
| 库的 `internal::routing` 中 | `super::normalize` | 从父模块 `internal` 找私有函数；子模块有访问权限 |
| 库的 `api` 中 | `crate::internal::routing::choose` | 仍在同一 library crate，能访问 `pub(crate)` 函数 |
| 启动 binary 中 | `gateway_demo::api::Gateway` | 从依赖的 library crate 找类型 |
| 检查 binary 中 | `gateway_demo::api::Gateway` | 同样使用库的公开 API |
| 任一 binary 中 | `crate::api::Gateway` | 本例无此路径：`crate::` 指该 binary，它没有声明 `api` 模块 |

库的 `api` 模块若用 `use` 引入 `choose`，后续可以用短名 `choose`；binary 若用 `use` 引入 `Gateway`，后续可以用短名 `Gateway`。两者都只是简写，权限不变。

因此，binary 不能靠 `use` 引入库里的 `choose` 来绕过 `pub(crate)`。`Cargo.toml` 负责依赖配置，`use` 不负责下载或添加依赖。

### 5. 改善对外入口：`pub use`

假设库根公开重新导出 `api` 中的 `Gateway`：

```text
原路径：gateway_demo::api::Gateway
                     │
                  pub use
                     ↓
新入口：gateway_demo::Gateway
```

| 场景 | 结果 |
|---|---|
| `api` 保持公开 | 原路径和新入口都能使用 |
| `api` 改为私有，但保留根上的公开重新导出 | 外部使用新入口，不能沿原 `api` 路径访问 |
| 使用新入口访问 `Gateway` 的私有 `name` 字段 | 仍然不允许，重新导出不改变字段权限 |
| 尝试公开重新导出仅为 `pub(crate)` 的 `choose` | 不允许把受限项提升为对外公开项 |

重新导出的是同一个类型，没有创建另一份 `Gateway`。它让调用方不必依赖内部目录组织。[Rust：可见性与重新导出](https://doc.rust-lang.org/reference/visibility-and-privacy.html)

### 6. 换成 Java 9：同一个设计如何理解

假设 Java 版本把库定义为 `gateway.core`，启动程序和检查程序分别属于 `gateway.app` 与 `gateway.inspect` 两个 JPMS module。

```text
module gateway.core
├── package demo.gateway.api       ← exports
│   └── public class Gateway
└── package demo.gateway.internal  ← 不 exports
    └── public class Routing

module gateway.app     ── requires ──→ gateway.core
module gateway.inspect ── requires ──→ gateway.core
```

| Java 9 场景 | 结果 | Rust 对照 |
|---|---|---|
| `gateway.app` 读取 core，core 导出 api，`Gateway` 是 public | 可以访问 `Gateway` | binary 通过库的公开路径访问公开类型 |
| app 没有读取 core | 即使写 `import` 也不能直接使用该类型 | 名字引入不能代替依赖关系 |
| `Routing` 是 public，但 internal 包未导出 | core 内可跨包使用，app 与 inspect 不能直接使用 | 接近 `pub(crate)` 的内部 API 意图 |
| `Gateway` 的 `name` 字段是 private | 普通外部代码不能访问 | Rust 私有字段也隐藏，但私有范围是模块而非类 |
| api 导出了，但类型不是 public | 外部仍不能访问该顶层类型 | 公开模块不会自动公开里面的类型 |
| 导出 `demo.gateway.api` 包 | 类型路径仍是 `demo.gateway.api.Gateway` | `exports` 不像 `pub use` 那样建立短路径 |

再把三个 JPMS 专有机制放进这个例子：

| Java 9 机制 | 本例中的效果 | Rust 对照边界 |
|---|---|---|
| core 对 `java.logging` 使用 `requires transitive` | 读取 core 的 app、inspect 也能读取 `java.logging` | 传递可读性，不是给类型重新命名或重新导出路径 |
| core 只向 app 导出 internal 包，即限定 `exports ... to ...` | app 可访问其中 public 类型，inspect 不可访问 | Rust 的这些内部可见性修饰符不能点名某个外部 crate |
| core 向某个框架模块 `opens` internal 包 | 允许该框架深度反射访问；不会因此赋予 app 普通源码访问权限 | `pub` 不承担这类反射权限职责 |

这些 Java 规则见 [Java 9：依赖](https://docs.oracle.com/javase/specs/jls/se9/html/jls-7.html#jls-7.7.1)与[导出和开放包](https://docs.oracle.com/javase/specs/jls/se9/html/jls-7.html#jls-7.7.2)。Java module 名通常不出现在类型的完整限定名里；Rust 外部路径通常从 crate 名开始。

本例的三个检查问题：**数构建单位看 package，数编译树看 crate，判断能否访问看路径和项的可见性。**

## 理解检查

1. Java 中 `public` 类型所在的包没有被导出，其他模块能直接使用它吗？Rust 中对应的访问障碍是什么？
2. 为什么 Java 子包没有父包的包访问权限，而 Rust 子模块能访问父模块的私有项？
3. 为什么 `main.rs` 不能仅凭“与 lib.rs 在同一个工程里”就访问库中的 `pub(crate)` 项？
4. Java `exports` 和 Rust `pub use` 对类型访问路径的影响有什么不同？
5. 在 `internal::xxx::routing::reset` 中，为什么 `pub(super)` 不允许 `internal` 直接访问？如何将范围精确扩大到 `internal` 子树？
