# 所有权、借用与释放

[总入口](../../index.md) · [问题索引](../../questions.md)

按表格顺序阅读；遇到具体问题也可以单独进入。前置指概念依赖，不意味着必须提前实施后续功能。

| 文章 | 解决的问题 | 前置 | 源码入口 |
|---|---|---|---|
| [Move 与部分移动](move-and-partial-move.md) | 为什么未移动的字段有时还能用，而整个对象不能用？ | struct、enum、impl 与 Self | [Worker::new](../../../../src/worker.rs) |
| [共享借用与可变借用](borrowing.md) | 为什么 policy 是 &mut，而 workers 是共享切片？ | struct、enum、impl 与 Self | [Policy::select](../../../../src/policy.rs) |
| [self、mut self、&self 与 &mut self](method-receivers.md) | 调用方不写 let mut，为什么仍能调用 mut self 方法？ | struct、enum、impl 与 Self | [ManagedTask::cancel_and_join](../../../../src/gateway/tests/support.rs) |
| [Copy 与 Clone：复制值还是共享句柄](copy-and-clone.md) | Arc::clone 和 String::clone 的成本与效果相同吗？ | struct、enum、impl 与 Self | [CircuitState](../../../../src/circuit_breaker.rs) |
| [生命周期：引用不能比数据活得久](lifetimes.md) | guard 中的 &'a Worker 限制了什么？ | 共享借用与可变借用 | [InFlightGuard<'a>](../../../../src/worker.rs) |
| [Drop 与 RAII：离开作用域如何收尾](drop-and-raii.md) | 为什么提前 return 也会让在途计数减一？ | Move 与部分移动 | [impl Drop for InFlightGuard](../../../../src/worker.rs) |
