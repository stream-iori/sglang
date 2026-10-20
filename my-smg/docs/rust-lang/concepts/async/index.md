# Rust 异步机制

[总入口](../../index.md) · [问题索引](../../questions.md)

按表格顺序阅读；遇到具体问题也可以单独进入。前置指概念依赖，不意味着必须提前实施后续功能。

| 文章 | 解决的问题 | 前置 | 源码入口 |
|---|---|---|---|
| [Future 与 await：创建、暂停和继续](future-and-await.md) | 创建 Future 与执行它是同一时间吗？ | 共享借用与可变借用、Trait | [operation().await](../../../../src/gateway/retry.rs) |
| [async move：把哪些值交给 Future](async-capture.md) | move 进去的是对象还是引用，要看什么？ | 共享借用与可变借用、Trait | [operation 闭包](../../../../src/gateway.rs) |
