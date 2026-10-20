# String 与 str：拥有文本还是借用文本

[分类目录](index.md) · [总入口](../../index.md) · [问题索引](../../questions.md)

**String 拥有可增长的文本，&str 是借用视图；as_str、to_owned 与 into 的行为不同。**

| 阅读准备 | 内容 |
|---|---|
| 前置 | 无；可以直接阅读。 |
| 源码入口 | [src/worker.rs](../../../../src/worker.rs)：`Worker::id` |
| 阅读时机 | 按当前课程需要阅读，不据此跳过 learning 的阶段验收。 |

## 项目入口：节点拥有地址，调用者借用地址

摘自 `src/worker.rs`：

```rust
pub fn address(&self) -> &str {
    &self.address
}
```

Worker 的字段是 String，方法只借出文本视图，调用方不取得字段所有权。

| 表达式 | 类型 | 是否复制文本 |
|---|---|---|
| `worker.address()` | `&str` | 否 |
| `worker.address().to_owned()` | `String` | 是，创建独立拥有值 |
| `String::from("Alice")` | `String` | 是 |
| `"Alice".to_owned()` | `String` | 是 |

```text
worker.address：String ──拥有──> 地址文本
                                  ↑
address() 返回 &str ──借用─────────┘

.to_owned() → 复制内容 → 新的 String
```

`to_owned` 可以读作“得到拥有型的值”。这里与 `String::from` 效果相同；其他类型的 to_owned 要看相应实现。

## String as str vs borrow

**`url.as_str()` 与 `&url` 的区别**

前提：这里的 `url` 是 `String`，不是 `reqwest::Url`。

| 表达式 | 类型 | 借用什么 | 复制字符串内容吗 |
|---|---|---|---|
| `url.as_str()` | `&str` | 字符串内容的视图 | 不复制 |
| `&url` | `&String` | String 对象 | 不复制 |
| `url.clone()` | `String` | 不借用，创建独立的拥有者 | 复制 |

```rust
let url = String::from("http://127.0.0.1:3001/chat");
let text: &str = url.as_str();
let object: &String = &url;
```

```text
url: String ──拥有──> 字符串内容
      ▲                  ▲
      │                  │
 &url: &String      url.as_str(): &str
```

两者都不转移 url 的所有权；借用使用期间，原值必须仍然有效。

## 自查与关联

&String 和 &str 是同一种类型吗？

[struct、enum、impl 与 Self](struct-enum-and-impl.md) · [元组、元组结构体与 unit](tuples-and-unit.md) · [表达式、返回值与分号](expressions-and-semicolons.md)
