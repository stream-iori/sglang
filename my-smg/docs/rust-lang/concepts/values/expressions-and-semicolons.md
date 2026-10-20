# 表达式、返回值与分号

[分类目录](index.md) · [总入口](../../index.md) · [问题索引](../../questions.md)

**Rust 表达式会产生值；块末尾的分号会影响这个块最终得到什么。**

| 阅读准备 | 内容 |
|---|---|
| 前置 | 无；可以直接阅读。 |
| 源码入口 | [src/gateway.rs](../../../../src/gateway.rs)：`selected_idex` |
| 阅读时机 | 按当前课程需要阅读，不据此跳过 learning 的阶段验收。 |

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
[配置驱动的共享策略](../../walkthroughs/policy-selection.md)。

## 自查与关联

为什么 match 后加分号可能把结果变成 ()？

[struct、enum、impl 与 Self](struct-enum-and-impl.md) · [元组、元组结构体与 unit](tuples-and-unit.md) · [String 与 str：拥有文本还是借用文本](string-and-str.md)
