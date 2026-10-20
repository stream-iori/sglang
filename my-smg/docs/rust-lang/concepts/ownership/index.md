# 所有权、借用与释放

[总入口](../../index.md) · [问题索引](../../questions.md)

按[当前阶段](../../stage-reading.md)选择文章；目录用于回查。前置只要求对应基础部分，项目应用另补该节前置。

| 文章 | 解决的问题 | 前置 | 源码入口 |
|---|---|---|---|
| [Move 与部分移动](move-and-partial-move.md) | 为什么未移动的字段有时还能用，而整个对象不能用？ | [struct、enum、impl 与 Self](../values/struct-enum-and-impl.md) | [src/worker.rs](../../../../src/worker.rs)：`Worker::new` |
| [共享借用与可变借用](borrowing.md) | 为什么 policy 是 &mut，而 workers 是共享切片？ | [Move 与部分移动](move-and-partial-move.md)基础部分；项目应用另按节标题补前置。 | [src/policy.rs](../../../../src/policy.rs)：`Policy::select` |
| [self、mut self、&self 与 &mut self](method-receivers.md) | 调用方不写 let mut，为什么仍能调用 mut self 方法？ | [共享借用与可变借用](borrowing.md)基础部分；项目应用另按节标题补前置。 | [src/gateway/tests/support.rs](../../../../src/gateway/tests/support.rs)：`ManagedTask::cancel_and_join` |
| [Copy 与 Clone：复制值还是共享句柄](copy-and-clone.md) | Arc::clone 和 String::clone 的成本与效果相同吗？ | [Move 与部分移动](move-and-partial-move.md)基础部分；项目应用另按节标题补前置。 | [src/circuit_breaker.rs](../../../../src/circuit_breaker.rs)：`CircuitState` |
| [生命周期：引用不能比数据活得久](lifetimes.md) | guard 中的 &'a Worker 限制了什么？ | [共享借用与可变借用](borrowing.md) | [src/worker.rs](../../../../src/worker.rs)：`InFlightGuard<'a>` |
| [Drop 与 RAII：离开作用域如何收尾](drop-and-raii.md) | 为什么提前 return 也会让在途计数减一？ | [Move 与部分移动](move-and-partial-move.md) | [src/worker.rs](../../../../src/worker.rs)：`impl Drop for InFlightGuard` |
