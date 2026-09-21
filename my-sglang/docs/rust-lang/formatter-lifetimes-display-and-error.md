# Rust：`Formatter<'_>`、生命周期、`Display` 与 `Error`

## 一句话

```text
Formatter<'_>：临时借用一个输出缓冲区的格式化器
Display：定义一个值怎样显示成人类可读文本
Error：在 Debug + Display 之上提供统一错误接口和错误来源链
From：定义错误类型之间如何转换
?：成功时继续，失败时提前返回，并在需要时调用 From 转换
```

这些概念不是一件事，但会在错误实现中同时出现：

```rust
impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // 把 ConfigError 写入 f
    }
}

impl std::error::Error for ConfigError {}
```

## 先看全貌

```text
ConfigError 值
      │
      │ println!("{error}") / error.to_string()
      ▼
Display::fmt(&error, &mut Formatter)
      │
      │ write!(f, "...")
      ▼
Formatter 借用的输出缓冲区
      │
      ├─> String
      ├─> 标准输出
      ├─> 标准错误
      └─> 其他 fmt::Write 目标

ConfigError: Error
      │
      ├─> 可以当作 &dyn Error / Box<dyn Error> 使用
      ├─> 可以通过 source() 暴露底层原因
      └─> 可被通用错误处理代码接收
```

`Display` 决定文字长什么样，调用方决定文字写到哪里；`Error` 则让这个值进入标准错误抽象。

## `fmt::Formatter<'_>` 拆开看

| 部分 | 含义 |
|---|---|
| `fmt` | `std::fmt` 模块的简称 |
| `Formatter` | 保存格式选项和输出目标的类型 |
| `<'_>` | 存在一个生命周期参数，由编译器推断 |
| `&mut Formatter<'_>` | 在本次 `fmt` 调用期间独占借用格式化器 |

标准库的真实实现细节可能随版本演进，但核心结构可以简化理解为：

```rust
pub struct Formatter<'a> {
    options: FormattingOptions,
    buf: &'a mut dyn Write,
}
```

其中：

```text
options -> 宽度、精度、对齐、正负号等格式选项
buf     -> 实际接收文本的输出缓冲区
```

用户通常不直接构造 `Formatter`。`println!`、`format!`、`write!` 等格式化系统创建它，再把
`&mut Formatter` 交给 `Display::fmt` 或 `Debug::fmt`。

## 为什么 Formatter 需要生命周期

`Formatter` 不拥有输出缓冲区，它只借用缓冲区：

```text
输出缓冲区（所有者）
      │
      │ &'a mut
      ▼
Formatter<'a>
```

因此编译器必须保证：

```text
Formatter 的使用时间 <= 被借用缓冲区的存活时间
```

如果缓冲区已经销毁，Formatter 仍然存在，它内部的指针就会悬空。生命周期参数让这种情况在
编译期被拒绝。

生命周期主要描述引用之间的有效范围关系，不是在运行时增加计时器，也不会让值活得更久。

## 生命周期 `'a` 是什么

显式命名生命周期时，可以写：

```rust
fn render<'a>(formatter: &mut fmt::Formatter<'a>) -> fmt::Result {
    // ...
}
```

拆开看：

```text
<'a>                    声明一个生命周期参数
Formatter<'a>           Formatter 内部缓冲区借用满足 'a
&mut Formatter<'a>      当前函数临时可变借用这个 Formatter
```

`'a` 不是具体的秒数，也不表示“整个函数生命周期”。它是编译器用来比较引用有效范围的名字。

## `'_`：让编译器推断生命周期

在 `Display` 实现里通常写：

```rust
fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result
```

`'_` 表示：

```text
这里必须有一个生命周期
但当前代码不需要给它命名
请编译器根据调用关系推断
```

它不表示：

- 忽略生命周期；
- 生命周期无限长；
- `'static`；
- 创建一个新对象。

显式命名与占位写法表达的核心约束相同：

```rust
fn fmt<'a>(&self, f: &mut fmt::Formatter<'a>) -> fmt::Result
```

```rust
fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result
```

这里只需要把 Formatter 用于当前调用，没有把其中的引用返回或保存，所以 `'_` 更简洁。

## 这里其实有两层借用

```rust
f: &mut fmt::Formatter<'_>
```

概念上可以展开为：

```text
&'call mut Formatter<'buffer>
   │                    │
   │                    └─ Formatter 内部借用输出缓冲区的时间
   └─ fmt 方法临时借用 Formatter 的时间
```

通常满足：

```text
'call <= 'buffer
```

源码把外层 `&mut` 的生命周期省略，把内层 Formatter 的生命周期写成 `'_`，两者都由编译器
推断。

## 为什么是 `&mut Formatter`

写入会推进缓冲区状态：

```text
空缓冲区
   -> 写入 "duplicate worker id: "
   -> 写入 worker_id
   -> 格式化结束
```

所以格式化方法需要可变访问 Formatter。`&mut` 同时保证本次写入期间没有另一个写入者并发
修改同一个 Formatter。

它不表示 `fmt` 拥有 Formatter。调用结束后，格式化系统仍然拥有并继续管理它。

## `Display` 的真实职责

标准库 trait 的核心形状是：

```rust
pub trait Display {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result;
}
```

职责是：

```text
把 self 的人类可读表示写入 f
```

`Display` 不直接负责：

- 选择控制台还是字符串；
- 打开文件；
- 记录日志级别；
- 给错误加颜色；
- 决定进程退出码。

这些由调用方或更上层组件决定。

## 哪些操作会使用 Display

| 写法 | 使用的 trait | 输出目标 |
|---|---|---|
| `println!("{value}")` | `Display` | 标准输出 |
| `eprintln!("{value}")` | `Display` | 标准错误 |
| `format!("{value}")` | `Display` | 新建的 `String` |
| `value.to_string()` | `Display`，通过 `ToString` | 新建的 `String` |
| `println!("{value:?}")` | `Debug` | 标准输出 |
| `format!("{value:#?}")` | `Debug` | 新建的 `String` |

实现 `Display` 后，标准库为该类型提供 `ToString` 的通用实现。因此通常不需要自己实现
`ToString`。

## `Debug` 与 `Display`

```text
Debug
├─ 面向开发者
├─ 常展示 enum variant 和字段结构
└─ 可以 derive

Display
├─ 面向用户、CLI、日志摘要
├─ 由类型作者决定简洁消息
└─ 通常手写或由 thiserror 生成
```

以配置错误为例：

```text
Debug:
InvalidWorkerAddress("worker-a", "not-a-url")

Display:
invalid address for worker worker-a: not-a-url
```

`Display` 文本不应自动被当作稳定的机器协议。程序需要分类错误时，应匹配 enum variant，而
不是解析显示字符串。

## `write!` 为什么返回 `fmt::Result`

`fmt::Result` 是格式化模块使用的结果类型，概念上等价于：

```rust
Result<(), fmt::Error>
```

```text
Ok(())          文本写入成功，没有额外值需要返回
Err(fmt::Error) 输出目标拒绝或无法完成写入
```

`write!(f, "...")` 已经返回 `fmt::Result`，所以一个 match 分支可以直接把它作为结果返回。

格式化失败不是业务错误，因此 `fmt` 方法一般只传播 `fmt::Error`，不要在这里执行网络请求、
复杂校验或产生其他副作用。

## 类型推断在这里做了什么

Rust 会从上下文推断多种信息，但不同占位符含义不同：

| 写法 | 推断什么 |
|---|---|
| `Formatter<'_>` | 生命周期参数 |
| `Vec::<usize>::new()` | 显式指定泛型类型，不是推断 |
| `let values: Vec<usize> = Vec::new()` | 从变量类型推断 `Vec::new()` 的元素类型 |
| `serde_json::from_str(input)` | 从函数返回类型推断目标类型 |
| `Self::EmptyWorkers` | 从当前 `impl` 推断 `Self` 是哪个类型 |

例如：

```rust
fn from_json(input: &str) -> Result<GatewayConfig, serde_json::Error> {
    serde_json::from_str(input)
}
```

`from_str` 是泛型函数，但返回类型要求 `GatewayConfig`，所以编译器推断目标类型就是
`GatewayConfig`。

推断不是“运行时猜测”。所有结果都必须在编译期唯一确定，否则编译器会要求补充类型或生命
周期信息。

## 生命周期省略与 `'_` 的区别

```rust
fn id(&self) -> &str
```

这里没有写生命周期，因为生命周期省略规则能判断返回引用来自 `self`。概念上接近：

```rust
fn id<'a>(&'a self) -> &'a str
```

而 `Formatter<'_>` 中，`Formatter` 这个类型本身声明了一个生命周期参数。写出 `'_` 是明确
告诉读者和编译器：“这个泛型参数是生命周期，请在这里推断”。

```text
省略生命周期 -> 语法规则自动补全
'_           -> 显式放置一个匿名生命周期占位符
```

两者都依赖编译器推断，但出现位置和表达目的不同。

## `'static` 不等于“这个引用永远存在”

`Error::source` 的常见签名是：

```rust
fn source(&self) -> Option<&(dyn Error + 'static)>
```

这里有两个不同概念：

```text
返回的 & 引用
└─ 生命周期省略后与 &self 绑定，不会超过 self

dyn Error + 'static
└─ trait object 内的具体错误不能借用短生命周期数据
```

因此 `+ 'static` 不是说返回的引用可以永远使用。它约束的是被引用错误对象内部不能依赖短期
借用。返回引用仍然受当前错误对象的生命周期限制。

拥有 `String`、`Vec` 等数据的错误通常可以满足 `'static`；保存 `&str` 的错误则可能需要额外
生命周期参数，且未必满足这一约束。

## `std::error::Error` 的核心结构

标准库中的关系可以简化为：

```rust
pub trait Error: Debug + Display {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        None
    }
}
```

`Debug + Display` 是 supertrait 约束：一个类型要实现 `Error`，必须先能用两种方式显示。

```text
ConfigError
├─ Debug   开发者格式
├─ Display 用户可读格式
└─ Error   标准错误接口
```

## 空的 `impl Error` 为什么有意义

```rust
impl std::error::Error for ConfigError {}
```

虽然实现体为空，但它明确承诺：

- 该类型满足标准错误契约；
- 可以作为 `dyn Error` 使用；
- 可以交给要求 `E: Error` 的泛型 API；
- `source()` 使用默认实现，返回 `None`。

当前 `ConfigError` 的三个 variant 都直接保存业务信息，没有包裹底层错误，所以 `None` 合理。

以后如果增加：

```text
ConfigError::Json(serde_json::Error)
```

就可以让 `source()` 返回内部的 `serde_json::Error`，形成错误原因链。

## 实现 Error 不会自动做什么

实现 `Error` 不会自动：

- 打印到控制台；
- 写日志；
- panic；
- 重试；
- 给进程设置退出码；
- 把任意错误转换成 `ConfigError`；
- 让所有 `?` 自动通过。

它提供的是统一接口，不是错误处理策略。

## 为什么需要统一 Error 接口

### 泛型代码

```rust
fn report<E: std::error::Error>(error: &E) {
    eprintln!("{error}");
}
```

调用方可以传入不同具体错误，函数只依赖 `Error` 契约。

### 动态分发

```rust
fn report(error: &dyn std::error::Error) {
    eprintln!("{error}");
}
```

调用时通过 trait object 的虚表找到具体 `Display` / `Error` 实现。trait object 原理见
[Trait、静态分发、动态分发与智能指针](traits-dispatch-and-smart-pointers.md)。

### 多种错误的统一返回

```rust
fn run() -> Result<(), Box<dyn std::error::Error>> {
    // 可以向上传播多种满足要求的错误
    Ok(())
}
```

`Box` 拥有具体错误，`dyn Error` 擦除具体类型。适合应用边界或快速组合；领域库通常更适合返回
明确的错误 enum，让调用者可以匹配具体 variant。

## `Error`、`From` 和 `?` 的分工

```text
Error
└─ 这个值符合标准错误接口

From<LowerError>
└─ LowerError 怎样转换成当前错误

?
├─ Ok(value) -> 取出 value，继续
└─ Err(error) -> 必要时用 From 转换，然后提前返回
```

仅实现 `Error`，不会自动获得错误转换。例如：

```text
serde_json::Error
        │ 需要 From<serde_json::Error>
        ▼
ConfigError
```

## `From` 与 `source()` 不要混淆

结论：`From` 用在错误产生和向上传播时，`source()` 用在错误产生后追查原因时。

| 对比 | `From` | `source()` |
|---|---|---|
| 方向 | 底层错误 → 外层错误 | 外层错误 → 查看底层错误 |
| 所有权 | 移动并包装错误 | 只返回共享引用 |
| 主要调用者 | `?` 或显式 `ConfigLoadError::from(...)` | 错误报告器、日志或诊断代码 |
| 是否改变错误类型 | 是 | 否 |
| 没有底层原因时 | 不适用 | 返回 `None` |

以配置加载为例：

```text
错误发生时：From

serde_json::Error
        │ move
        ▼
ConfigLoadError::Parse(error)

错误报告时：source()

ConfigLoadError::Parse(error)
        │ borrow
        ▼
&serde_json::Error
```

转换实现：

```rust
impl From<serde_json::Error> for ConfigLoadError {
    fn from(error: serde_json::Error) -> Self {
        Self::Parse(error)
    }
}
```

原因链实现：

```rust
impl std::error::Error for ConfigLoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Parse(error) => Some(error),
            Self::Validation(error) => Some(error),
        }
    }
}
```

两者连接成完整流程：

```text
1. serde_json 产生错误
2. ? 使用 From，把它包装成 ConfigLoadError::Parse
3. 调用方收到 ConfigLoadError
4. 诊断代码通过 source() 查看原始 serde_json::Error
```

一句话：

```text
From 负责“装进去”，source() 负责“看里面”。
```

`?` 的完整控制流见[`Result`、`?` 与错误传播](result-question-mark-and-error-propagation.md)。

## 当前 ConfigError 的完整角色

```text
ConfigError enum
├─ EmptyWorkers
├─ DuplicateWorkerId(String)
└─ InvalidWorkerAddress(String, String)
          │
          ├─ derive(Debug)       -> {:?}
          ├─ impl Display       -> {} / to_string()
          └─ impl Error         -> dyn Error / source()
```

错误 variant 使用 `String` 而不是 `&str`，因此错误拥有自己的数据：

```text
GatewayConfig 被释放
        │
        └─ ConfigError 仍能独立存在和向上传播
```

如果错误借用配置中的 `&str`，`ConfigError` 本身也需要生命周期参数，并且传播范围会被原配置
的生命周期限制。

## 常见编译错误

### `missing lifetime specifier`

编译器无法判断返回引用来自哪个输入。应先画出“谁拥有数据、谁借用数据”，再补生命周期，
不要机械添加 `'static`。

### `Formatter` 缺少生命周期参数

`Formatter` 是带生命周期参数的类型。实现格式化 trait 时通常使用 `Formatter<'_>`。

### `ConfigError` 没有实现 Display

常见表现是不能使用 `{}` 或不能实现 `Error`。先实现 `Display`，或者使用 `thiserror` 生成。

### `?` 无法转换错误

检查当前函数的错误类型与下层错误类型之间是否存在 `From` 实现。`Error` trait 本身不负责
转换。

### `source()` 返回的引用生命周期不够长

检查内部错误是否借用了短期数据。不要通过泄漏内存或强行标记 `'static` 绕过；优先让错误
拥有必要的上下文数据。

## 推荐阅读顺序

```text
所有权、转移与借用
        │
        v
本文：Formatter、生命周期、Display、Error
        │
        ├─> Result、?、错误传播
        ├─> 常见 derive / thiserror
        └─> trait 静态与动态分发
```

这些机制背后的分层、信息保留、所有权与关注点分离原则，见
[Rust 软件设计的第一性原理](desgin-principle.md)。

## 一句话记忆

```text
生命周期：证明借用不会比数据活得更久。
Formatter：借用输出目标，携带格式选项。
Display：决定人类可读文字长什么样。
Error：把具体错误接入统一标准接口。
From：定义错误怎么转换。
?：沿失败路径提前返回。
```
