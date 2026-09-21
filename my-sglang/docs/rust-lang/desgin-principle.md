# Rust 软件设计的第一性原理

## 结论

```text
错误是数据，不是字符串。
底层保留事实，上层增加业务语义。
拥有者负责保存，借用者只负责查看。
机制负责传递，业务边界负责决策。
```

本文不重复 Rust 语法，使用 `my-smg` 的配置加载作为统一例子。

## 统一例子：配置加载流水线

```text
JSON 文本
   │ 解析
   ├────────失败────> serde_json::Error
   ▼
GatewayConfig
   │ 校验
   ├────────失败────> ConfigError
   ▼
可运行配置

高层统一入口：
ConfigLoadError
├─ Parse(serde_json::Error)
└─ Validation(ConfigError)
```

| 类型 | 负责什么 |
|---|---|
| `serde_json::Error` | JSON 语法和字段类型错误 |
| `ConfigError` | 空节点、重复 ID、非法地址等业务错误 |
| `ConfigLoadError` | 为完整配置加载提供统一错误边界 |

## 原则总表

| 第一性原理 | 当前例子 |
|---|---|
| 失败也是数据 | 使用 `Result<T, E>` |
| 阶段不同，语义不同 | Parse 与 Validation 分开 |
| 底层具体，高层统一 | 两种底层错误包装为 `ConfigLoadError` |
| 增加语义，不丢证据 | 包装原错误，而不是只保存错误字符串 |
| 所有权跟随责任 | 外层错误拥有内部错误，`source()` 只借用 |
| 机制与策略分离 | `?` 传播，`main` 决定打印或退出 |
| 机器结构与人类文字分离 | enum variant 给程序，`Display` 给人 |
| 失败尽早返回 | 解析失败后不校验，校验失败后不启动 |
| 复杂度按需求引入 | 没有共享需求就不提前使用 `Arc`、锁或 `dyn` |

## 1. 失败也是数据

一次可能失败的操作只有两类结果：

```text
操作
├─ 成功：T
└─ 失败：E
```

Rust 用类型明确表达它：

```rust
Result<T, E>
```

这让失败出现在函数签名中，调用者不能假装失败路径不存在。

```text
隐藏失败：返回特殊值、只写日志、突然 panic
明确失败：Result<GatewayConfig, ConfigLoadError>
```

详细控制流见[`Result`、`?` 与错误传播](result-question-mark-and-error-propagation.md)。

## 2. 阶段不同，错误语义不同

同一句“配置错了”可能对应不同事实：

| 输入 | 解析 | 校验 | 错误阶段 |
|---|---:|---:|---|
| 缺少 `}` | 失败 | 不执行 | Parse |
| `workers` 是字符串 | 失败 | 不执行 | Parse |
| `workers` 是空数组 | 成功 | 失败 | Validation |
| Worker ID 重复 | 成功 | 失败 | Validation |

原则：

```text
先问“在哪一步失败”，再设计错误类型。
```

如果所有错误都叫 `InvalidConfig`，调用者很难给出准确提示，也难以决定是否重试或修复输入。

## 3. 底层具体，高层统一

底层函数返回它最了解的错误：

```text
from_json() -> serde_json::Error
validate()  -> ConfigError
```

完整流程再统一：

```text
from_json_validated() -> ConfigLoadError
```

```text
底层具体
├─ 信息精确
└─ 容易单独测试

高层统一
├─ 调用方式简单
└─ 提供稳定业务边界
```

不要在所有底层函数中直接返回 `Box<dyn Error>`。类型擦除虽然统一，但会降低调用方按错误种类
做决策的能力。

## 4. 增加语义，但不要丢失证据

错误可以“转换成字符串”，也可以“包装原错误”：

```text
不推荐：
serde_json::Error
   -> "parse failed"
   -> 行号、列号、原始原因丢失

推荐：
serde_json::Error
   -> ConfigLoadError::Parse(error)
   -> 原始错误仍完整保留
```

`From` 增加外层语义：

```text
serde_json::Error
        │ From
        ▼
ConfigLoadError::Parse(error)
```

`source()` 保留回看底层证据的路径：

```text
ConfigLoadError::Parse(error)
        │ source()
        ▼
&serde_json::Error
```

一句话：

```text
From 负责装进去，source() 负责看里面。
```

技术细节见[`From` 与 `source()` 不要混淆](formatter-lifetimes-display-and-error.md#from-与-source-不要混淆)。

## 5. 所有权跟随责任

谁负责让数据继续存在，谁就应该拥有它：

```text
ConfigLoadError
└─ 拥有内部 serde_json::Error

错误报告器
└─ 只通过 source() 借用内部错误
```

| 场景 | 选择 |
|---|---|
| 错误要脱离原配置继续传播 | 保存 `String` |
| 只在当前调用中查看错误原因 | 返回 `&dyn Error` |
| 临时读取节点列表 | 使用 `&[Worker]` |
| 调用后仍需保留对象 | 不按值消费它 |

这也是 `DuplicateWorkerId(String)` 比借用 `&str` 更适合当前错误类型的原因：配置销毁后，错误
仍然可以独立存在。

所有权基础见[所有权、转移与借用](ownership-move-and-borrowing.md)。

## 6. 机制与策略分离

每个组件只回答一个问题：

| 组件 | 回答的问题 |
|---|---|
| `Display` | 这个错误怎样显示成人类可读文字？ |
| `Error` | 它能否进入标准错误接口，底层原因是谁？ |
| `From` | 下层错误怎样包装成上层错误？ |
| `?` | 失败时怎样快速返回？ |
| 日志层 | 何时、以什么级别记录？ |
| `main` / HTTP 边界 | 打印、退出还是转换成状态码？ |

反例：在 `validate()` 中直接打印错误。

```text
validate 同时负责校验和输出
        │
        ├─ 测试难控制输出
        ├─ 库代码替调用方做决定
        └─ HTTP、CLI、测试无法选择不同处理方式
```

更好的方式是让 `validate()` 返回结构化错误，由最外层决定如何展示。

## 7. 机器结构与人类文字分离

程序判断错误时匹配结构：

```text
ConfigError::EmptyWorkers
ConfigError::DuplicateWorkerId(id)
```

人类阅读错误时使用 `Display`：

```text
at least one worker is required
duplicate worker id: worker-a
```

```text
程序逻辑  -> enum variant
用户界面  -> Display 文本
开发调试  -> Debug 文本
```

不要让程序通过搜索错误字符串来分类错误。文案可以改变，结构化 variant 才是稳定的程序接口。

`Debug`、`Display`、`Error` 的关系见
[`Formatter<'_>`、生命周期、`Display` 与 `Error`](formatter-lifetimes-display-and-error.md)。

## 8. 失败尽早返回

配置加载具有依赖顺序：

```text
解析成功，才有配置可校验
校验成功，才有配置可启动
```

因此正确流程是：

```text
parse()?
   │ 成功
   ▼
validate()?
   │ 成功
   ▼
start
```

任一步失败都立即结束当前流程：

```text
解析失败 -> 不运行校验
校验失败 -> 不启动服务
```

这不是为了少写代码，而是防止后续步骤处理不存在或不可信的数据。

## 9. 复杂度按真实需求引入

抽象和并发工具都有成本：

| 工具 | 只有什么时候才引入 |
|---|---|
| trait | 已经出现多个需要统一调用的实现 |
| `dyn Trait` | 需要运行时切换具体实现 |
| `Box` | 需要独占拥有动态大小值 |
| `Arc` | 确实需要多个长期所有者 |
| `Mutex` / 原子类型 | 确实存在共享可变状态或并发更新 |
| 统一错误 enum | 高层流程需要组合多种错误 |

原则：

```text
先出现问题，再引入解决该问题的抽象。
```

提前照搬上游的 `Arc`、原子变量和复杂 trait，会隐藏当前阶段真正要学习的所有权与行为边界。

## 设计选择速查

| 问题 | 推荐选择 | 原因 |
|---|---|---|
| 需要让程序区分错误种类 | error enum | 可穷举匹配 |
| 需要给用户显示错误 | `Display` | 人类可读 |
| 需要保留底层原因 | 包装原错误并实现 `source()` | 不丢证据 |
| 需要 `?` 转换错误 | 实现 `From` | 明确转换方向 |
| 错误必须独立于输入存在 | 拥有 `String` 等数据 | 不受输入生命周期限制 |
| 应用边界临时统一任意错误 | `Box<dyn Error>` | 使用方便，但类型信息被擦除 |
| 库或领域层需要精确处理 | 具体错误 enum | 保留结构化语义 |
| 测试中的固定输入必须成功 | `expect("原因")` | 失败信息清楚 |
| 用户输入可能失败 | `Result` + `?` | 不因普通输入错误 panic |

## 设计检查清单

设计一个可能失败的新流程时，依次回答：

1. 哪一步可能失败？
2. 每种失败需要保留哪些事实？
3. 哪一层最了解底层细节？
4. 哪一层需要统一调用边界？
5. 错误应该拥有数据，还是只临时借用？
6. 程序怎样分类错误，人类怎样阅读错误？
7. 是否保留了底层 source？
8. `?` 需要哪些 `From` 转换？
9. 当前抽象解决了真实问题，还是提前模仿复杂系统？

## 一句话总结

```text
用类型表达失败，
用分层表达语义，
用所有权保证错误能活多久，
用 From 保留并包装原因，
用 source() 提供诊断路径，
最后由系统边界决定怎么处理。
```
