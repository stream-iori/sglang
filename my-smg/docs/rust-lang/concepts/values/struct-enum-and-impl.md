# struct、enum、impl 与 Self

[分类目录](index.md) · [总入口](../../index.md) · [问题索引](../../questions.md)

**struct 装数据，enum 表示不同状态，impl 定义操作，Self 指当前实现的类型。**

| 阅读准备 | 内容 |
|---|---|
| 前置 | 无；可以直接阅读。 |
| 源码入口 | [src/worker.rs](../../../../src/worker.rs)：`Worker / HealthStatus / impl Worker` |
| 阅读时机 | 按当前课程需要阅读，不据此跳过 learning 的阶段验收。 |

## 从项目的节点看起

下面是 `src/worker.rs` 的节选；省略了其他字段和实现：

```rust
pub enum HealthStatus {
    Healthy,
    Unhealthy,
}

pub struct Worker {
    id: String,
    address: String,
    // 其他字段省略
}
```

```text
Worker：一种节点对象的类型
├── id：节点标识
└── address：节点地址

HealthStatus：状态只能在定义的变体中选择
├── Healthy
└── Unhealthy
```

`struct` 类似 Java 对象里的数据结构，但方法在 `impl` 块中声明；它不是类继承机制。
`enum` 的变体还可以带数据，例如配置错误中的 `DuplicateWorkerId(String)`。

项目中的方法节选：

```rust
pub fn id(&self) -> &str {
    &self.id
}
```

| 写法 | 含义 |
|---|---|
| `Worker` | 类型名 |
| `worker` | 保存具体值的变量名 |
| `impl Worker` | 为 Worker 定义操作 |
| `Self` | 当前实现中的类型，此处是 Worker |
| `self` | 本次操作的接收者 |
| `Worker::new(...)` | 通过类型调用不含 self 的关联函数 |
| `worker.id()` | 对具体对象调用方法 |

构造函数叫 `new` 是惯例；返回 `Self` 表示返回 Worker。它是否分配堆内存取决于内部字段，不能把调用 new 等同于 Java 的对象分配模型。

## 自查与关联

Worker::new 与 worker.id 分别如何调用？

[元组、元组结构体与 unit](tuples-and-unit.md) · [表达式、返回值与分号](expressions-and-semicolons.md) · [String 与 str：拥有文本还是借用文本](string-and-str.md)
