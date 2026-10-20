# 元组、元组结构体与 unit

[分类目录](index.md) · [总入口](../../index.md) · [问题索引](../../questions.md)

**圆括号可以表示组合值、unit 或函数调用，要结合所在位置读。**

| 阅读准备 | 内容 |
|---|---|
| 前置 | 无；可以直接阅读。 |
| 源码入口 | [src/gateway/retry.rs](../../../../src/gateway/retry.rs)：`type HttpResponse` |
| 阅读时机 | 按当前课程需要阅读，不据此跳过 learning 的阶段验收。 |

## 项目中的响应元组

```rust
type HttpResponse = (StatusCode, String);
```

这是项目的别名声明，`HttpResponse` 包含两个有顺序的值：

```text
HttpResponse
├── .0：StatusCode
└── .1：String
```

`(status, body)` 是元组值，`let (status, body) = response` 是解构模式。
与元组不同，`pub struct State<S>(pub S);` 是有名字的元组结构体：外层 pub 公开类型，内层 pub 公开第 0 个字段，S 是字段类型。

| 写法 | 位置与含义 |
|---|---|
| `()` | 类型位置：unit 类型，不携带额外信息 |
| `()` | 表达式位置：unit 的唯一值 |
| `(42,)` | 只有一个元素的元组；逗号不能省略 |
| `(42)` | 整数表达式加括号，不是元组 |
| `pending::<()>()` | 尖括号里的 () 是类型，末尾 () 是空参数调用 |
| `Ok(())` | Result 成功，成功值不携带额外数据 |

`src/gateway/tests/support.rs` 中，`TestServer` 保存 `ManagedTask<()>`：服务器任务完成时不产出业务值。unit 不是“这个函数永远不返回”；永不完成是 pending 的行为。

## 自查与关联

() 与 (StatusCode, String) 携带多少信息？

[struct、enum、impl 与 Self](struct-enum-and-impl.md) · [表达式、返回值与分号](expressions-and-semicolons.md) · [String 与 str：拥有文本还是借用文本](string-and-str.md)
