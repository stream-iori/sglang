# 模式与控制流

[总入口](../../index.md) · [问题索引](../../questions.md)

按表格顺序阅读；遇到具体问题也可以单独进入。前置指概念依赖，不意味着必须提前实施后续功能。

| 文章 | 解决的问题 | 前置 | 源码入口 |
|---|---|---|---|
| [match 分支与守卫的执行顺序](match-and-guards.md) | 两个 Err(error) 为什么不会都执行？ | struct、enum、impl 与 Self | [TestServer::stop](../../../../src/gateway/tests/support.rs) |
| [if let 与 let else：成功值在哪里可用](if-let-and-let-else.md) | else 中为什么不能使用这次绑定的 index？ | struct、enum、impl 与 Self | [let Some(index)](../../../../src/gateway.rs) |
| [解构：从 State、结构体和元组中取值](destructuring.md) | State 是结构体、表达式还是模式？ | struct、enum、impl 与 Self | [State(state): State<Arc<GatewayState>>](../../../../src/gateway.rs) |
| [ref、引用模式与解引用](reference-patterns.md) | 为什么 Drop 里只借用 handle 而不移走？ | struct、enum、impl 与 Self | [if let Some(handle) = &self.handle](../../../../src/gateway/tests/support.rs) |
