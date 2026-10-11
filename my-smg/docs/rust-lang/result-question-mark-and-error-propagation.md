# Rust：`Result`、`?` 与错误传播

## 一句话

```text
Result<T, E>：一次操作要么成功得到 T，要么失败得到 E。
?：成功时取出 T 并继续；失败时让当前函数提前返回错误。
```

“把错误交给调用者”不是忽略错误，而是当前函数不在这里处理它，将失败信息继续向上一层
返回。

错误分层、fail-fast 和“底层具体、高层统一”的设计原因，见
[Rust 软件设计的第一性原理](desgin-principle.md)。

## 从 JSON 配置解析开始

下面的关联函数尝试把 JSON 文本解析成 `GatewayConfig`：

```rust
impl GatewayConfig {
    pub fn from_json(input: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(input)
    }
}
```

当前 `impl GatewayConfig` 中，`Self` 就是 `GatewayConfig`，所以返回类型等价于：

```rust
Result<GatewayConfig, serde_json::Error>
```

它有两种可能：

```text
Ok(GatewayConfig)      JSON 有效，成功得到配置
Err(serde_json::Error) JSON 语法或字段类型不符合要求
```

这里不需要 `?`，因为 `serde_json::from_str(input)` 返回的类型正好就是当前函数要返回的类型，
直接返回最简洁。

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

## `?` 的核心行为

假设一个函数内部调用了解析操作：

```rust
fn load_config(input: &str) -> Result<GatewayConfig, serde_json::Error> {
    let config = GatewayConfig::from_json(input)?;
    Ok(config)
}
```

关键语句是：

```rust
let config = GatewayConfig::from_json(input)?;
```

执行结果分成两条路径：

```text
GatewayConfig::from_json(input)
          │
          ├─ Ok(config)  -> ? 取出 config -> 继续执行下一行
          │
          └─ Err(error)  -> 当前函数立即返回 Err(error)
```

所以“`?` 不会继续执行后面的代码”只针对 `Err` 分支。遇到 `Ok` 时，后续代码会正常执行。

## `?` 近似等价于什么

上面的代码可以展开为：

```rust
fn load_config(input: &str) -> Result<GatewayConfig, serde_json::Error> {
    let config = match GatewayConfig::from_json(input) {
        Ok(config) => config,
        Err(error) => return Err(error),
    };

    Ok(config)
}
```

重点是 `return Err(error)`：

- `return` 结束的是当前的 `load_config` 函数；
- 它不会自动结束整个程序；
- `load_config` 的调用者会收到这个 `Err`；
- 调用者可以处理它，也可以再次使用 `?` 继续向上传播。

真实的 `?` 还支持通过 `From` / `Into` 转换错误，因此“等价于 match”是帮助理解控制流的
近似模型，不是其全部实现细节。

## “交给调用者”到底是什么意思

假设调用关系是：

```text
main
  └─ start_gateway
       └─ load_config
            └─ GatewayConfig::from_json
```

如果最底层解析失败：

```text
from_json 返回 Err
       │
       │ load_config 使用 ?
       ▼
load_config 提前返回 Err
       │
       │ start_gateway 也使用 ?
       ▼
start_gateway 提前返回 Err
       │
       ▼
main 最终决定如何打印错误以及用什么退出码结束
```

每一层都可以选择：

1. 当场处理并恢复；
2. 把底层错误转换为更符合本层语义的错误；
3. 使用 `?` 继续交给上一层。

因此，`?` 表达的是“这层没有足够上下文恢复，请上层决定”，而不是“错误不重要”。

## 为什么当前函数也必须允许返回错误

下面的函数声明返回普通 `GatewayConfig`：

```rust
fn load_config(input: &str) -> GatewayConfig
```

它无法直接使用一个可能产生 `serde_json::Error` 的 `?`，因为函数签名没有错误返回通道。

通常应把签名改成 `Result`：

```rust
fn load_config(input: &str) -> Result<GatewayConfig, serde_json::Error>
```

这样 `Err` 才能从当前函数返回给调用者。

看到类似编译错误时：

```text
the `?` operator can only be used in a function that returns `Result` or `Option`
```

应检查当前函数的返回类型，而不是简单地删除 `?` 或改成 `unwrap()`。

## 为什么成功值需要重新包进 `Ok`

使用 `?` 后，成功分支的 `Result` 外壳被打开：

```rust
let config = GatewayConfig::from_json(input)?;
// config: GatewayConfig，不是 Result<GatewayConfig, _>
```

如果当前函数声明返回 `Result<GatewayConfig, _>`，最终需要重新包装：

```rust
Ok(config)
```

类型变化是：

```text
Result<GatewayConfig, Error>
            │ ?
            ▼
GatewayConfig
            │ Ok(...)
            ▼
Result<GatewayConfig, Error>
```

之所以先取出再包装，通常是因为中间还需要校验、转换或执行其他可能失败的操作。如果什么都
不做，直接返回原来的 `Result` 更简单。

## 直接返回与使用 `?` 的选择

### 直接返回

```rust
fn from_json(input: &str) -> Result<Self, serde_json::Error> {
    serde_json::from_str(input)
}
```

适合内部调用的返回类型与当前函数完全相同，并且没有后续处理。

### 使用 `?`

```rust
fn from_json(input: &str) -> Result<Self, serde_json::Error> {
    let config = serde_json::from_str(input)?;
    // 在这里继续处理 config
    Ok(config)
}
```

适合成功后还需要继续执行校验、转换或其他步骤。

不要为了使用 `?` 而机械地拆开再包装；它的价值是简化多步骤失败控制流。

## 多个步骤如何连续传播错误

一个真实的配置加载流程可能包含：

```text
读取文件 -> 解析 JSON -> 校验业务规则 -> 构造运行时对象
```

伪代码可以写成：

```rust
fn load(path: &Path) -> Result<GatewayConfig, ConfigError> {
    let text = read_file(path)?;
    let config = parse_json(&text)?;
    validate(&config)?;
    Ok(config)
}
```

任意一步失败都会提前返回，后面的步骤不会执行。例如读取失败时，不会继续解析；解析失败
时，也不会拿一个不存在的配置去校验。

这比多层嵌套 `match` 更容易看出“成功主线”。

## 错误类型不同时发生什么

文件读取可能返回 `std::io::Error`，JSON 解析可能返回 `serde_json::Error`，而函数也许统一返回
自定义的 `ConfigError`。

要让 `?` 自动传播，当前错误必须能够转换成函数声明的错误类型。概念上需要：

```text
ConfigError: From<std::io::Error>
ConfigError: From<serde_json::Error>
```

传播过程近似为：

```text
Err(serde_json::Error)
       │ From::from
       ▼
Err(ConfigError)
```

这就是 `thiserror` 中 `#[from]` 或手写 `From` 实现经常与 `?` 一起出现的原因。相关内容见
[常见 derive 速查中的 thiserror 章节](common-derive.md#thiserrorerror)。

`Error`、`Display`、`From` 和 `?` 各自负责什么，以及 `dyn Error` 如何统一不同错误，见
[`Formatter<'_>`、生命周期、`Display` 与 `Error`](formatter-lifetimes-display-and-error.md)。
其中 `From` 与 `source()` 的方向和所有权差异，见该文的
[`From` 与 `source()` 不要混淆](formatter-lifetimes-display-and-error.md#from-与-source-不要混淆)。

## 类型转换：Into、From、TryInto 与 TryFrom

`.into()` 来自标准库的 `Into` trait：把当前值转换成调用位置需要的目标类型。转换需要对应的
trait 实现；编译器根据变量标注、函数参数或返回类型推断目标。它不是任意类型之间的强制转换。

### 网关启动错误中的 `.into()`

下面是与 `my-smg` 启动错误相关的独立例子：

```rust
fn startup_error() -> Result<(), Box<dyn std::error::Error>> {
    let error = std::io::Error::new(
        std::io::ErrorKind::InvalidInput,
        "duplicate worker id",
    );

    Err(error.into())
}
```

```text
std::io::Error
       │ .into()
       ▼
Box<dyn std::error::Error>
       │ Err(...)
       ▼
函数的错误返回值
```

函数声明的错误类型决定了转换目标。这里会把具体错误装进 Box，通过 `dyn Error` 统一返回。
底层错误本身仍被保留，不是转换成错误字符串。若没有足够上下文，编译器可能无法推断目标类型。

### 常见场景与类型推断

| 场景 | 转换 | 目标类型来自哪里 |
|---|---|---|
| 创建拥有所有权的字符串 | `&str` → `String` | 变量标注或函数参数 |
| 统一错误返回类型 | 具体错误 → `Box<dyn Error>` | 函数的错误返回类型 |
| 自定义数据转换 | 源类型 → 目标类型 | 对应 trait 实现与调用上下文 |

```rust
let text: String = "hello".into();
```

这里 `String` 指定目标；从 `&str` 创建 String 会分配并复制文本。其他转换是否分配、复制或
只包装原值，要看具体实现，不能把 `.into()` 一概理解为 clone。

### From 与 Into：同方向转换的两个入口

```rust
let first: String = "hello".into();
let second = String::from("hello");
```

| 写法 | 从哪一端描述转换 | 实际方向 |
|---|---|---|
| `source.into()` | 源值：把我转换成目标类型 | 源 → 目标 |
| `Target::from(source)` | 目标类型：用源值创建我 | 源 → 目标 |

它们不是互相撤销的操作。实现 `From<Source> for Target` 后，标准库的通用实现会自动提供
`Into<Target> for Source`。通常优先实现 From；泛型接口只需允许调用 `.into()` 时，可以用
`Into<Target>` 作为约束。

### 所有权与可逆性

`Into::into(self)` 消费接收者。传入非 Copy 的拥有型值时，它通常会移动，原变量不能继续使用。
传入 `&str` 时，消费的是可复制的引用，不会取得或销毁被借用文本的所有权。

转换不保证可逆。例如 `String::as_str()` 只是借用 String 保存的文本，不会撤销之前的
`&str` → `String` 转换。值转换、借用和克隆的关系见
[所有权文章中的方法接收者](ownership-move-and-borrowing.md)。

### 可能失败的转换

| trait / 方法 | 返回类型 | 适用情况 |
|---|---|---|
| `From` / `from()` | 目标值 | 转换不通过 Result 报告失败 |
| `Into` / `into()` | 目标值 | 同上，从源值调用 |
| `TryFrom` / `try_from()` | `Result<目标值, 错误>` | 转换可能失败 |
| `TryInto` / `try_into()` | `Result<目标值, 错误>` | 同上，从源值调用 |

```rust
let value: Result<u8, _> = 300_u16.try_into();
assert!(value.is_err()); // 300 超出了 u8 的范围。
```

实现 TryFrom 会获得对应的 TryInto。From / Into 应用于语义合理、不丢失信息且不主动失败的
转换；需要检查范围或业务约束时，选择 TryFrom / TryInto。

### 与 `?`、Error 和 source 的关联

| 写法 / 能力 | 职责 |
|---|---|
| `error.into()` | 显式转换错误值，本身不提前返回 |
| `return Err(error.into())` | 转换错误，再结束当前函数 |
| `operation()?` | 成功取值；失败时必要的 From 转换并提前返回 |
| `Error` | 定义标准错误接口 |
| `source()` | 借用查看底层原因 |

错误接口、动态错误与转换契约见
[Error、From 和 ? 的分工](formatter-lifetimes-display-and-error.md#errorfrom-和--的分工)；
转换与原因链的区别见
[From 与 source 对照](formatter-lifetimes-display-and-error.md#from-与-source-不要混淆)。

## `?` 也能用于 `Option`

当函数返回 `Option<T>` 时，`?` 的行为类似：

```text
Some(value) -> 取出 value，继续执行
None        -> 当前函数立即返回 None
```

例如：

```rust
fn first_name(names: &[String]) -> Option<&str> {
    let first = names.first()?;
    Some(first.as_str())
}
```

`Result` 的 `?` 传播 `Err`，`Option` 的 `?` 传播 `None`。相关模式示例见
[`if let`、`ref` 与解引用](if-let-ref-and-deref.md#写法二返回值本来就是-option-时使用-)。

## `?` 与 `unwrap`、`expect` 的区别

| 写法 | 失败时行为 | 适合场景 |
|---|---|---|
| `?` | 把错误返回给调用者 | 正式业务流程中的错误传播 |
| `match` | 当前层明确处理每个分支 | 需要恢复、分类或自定义行为 |
| `unwrap()` | panic | 已有严格不变量，失败代表程序错误 |
| `expect("...")` | 带消息 panic | 测试，或失败确实不可能且需说明原因 |

测试有效输入时可以使用：

```rust
let config = GatewayConfig::from_json(input)
    .expect("valid config should parse");
```

因为测试目标就是“这份输入必须成功”，失败时 panic 能让测试立刻失败。但正式启动路径不能因
用户配置错误而随意 `unwrap()`，应返回清晰错误。

## 解析错误与业务校验错误不同

Serde 负责把数据转换成 Rust 类型，不自动理解所有业务规则：

| JSON 内容 | 反序列化 | 业务上是否一定有效 |
|---|---:|---:|
| 缺少右花括号 | 失败 | 否 |
| `workers` 是字符串 | 失败 | 否 |
| `workers` 是空数组 | 成功 | 不一定 |
| 两个 Worker ID 相同 | 成功 | 不一定 |
| 地址是普通字符串 `hello` | 成功 | 不一定 |

因此后续流程通常是：

```text
JSON 文本
   │ serde_json::from_str
   ▼
结构正确的 GatewayConfig
   │ validate
   ▼
业务上可用的 GatewayConfig
```

解析错误与校验错误可以最终统一为 `ConfigError`，但它们代表不同失败阶段，错误消息也应让
用户能够区分。

## 所有权发生了什么

```rust
let config = GatewayConfig::from_json(input)?;
```

- `input: &str` 只是被借用；
- 成功时，`GatewayConfig` 从 `Ok` 中被取出并移动到局部变量 `config`；
- 失败时，错误值被移动到当前函数的返回值中；
- 配置字段是 `String`、`Vec` 等拥有型数据，因此成功结果不继续借用输入文本。

所有权基础见[所有权、转移与借用](ownership-move-and-borrowing.md)。

## 常见误解

### `?` 会处理错误

不准确。`?` 通常只是传播错误。日志记录、重试、转换为 HTTP 状态码等处理仍需由合适的层
完成。

### `?` 遇到错误会退出整个程序

不一定。它只让当前函数提前返回。是否最终结束程序，取决于上层如何处理返回的错误。

### 使用 `?` 后返回类型变成成功类型

局部变量会得到成功类型，但当前函数仍按照声明返回 `Result` 或 `Option`。

### 所有错误都能自动用 `?` 转换

不正确。源错误必须与目标错误相同，或者存在合适的 `From` 转换。

### `?` 可以放在任意函数中

不正确。当前函数或闭包的返回类型必须支持相应的提前返回语义。

## 一句话记忆

```text
Ok  -> ? 拆开成功值，继续往下走。
Err -> ? 结束当前函数，把错误交给调用者。
```
