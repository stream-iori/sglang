# 类型、值与容器

[总入口](../../index.md) · [问题索引](../../questions.md)

按表格顺序阅读；遇到具体问题也可以单独进入。前置指概念依赖，不意味着必须提前实施后续功能。

| 文章 | 解决的问题 | 前置 | 源码入口 |
|---|---|---|---|
| [struct、enum、impl 与 Self](struct-enum-and-impl.md) | Worker::new 与 worker.id 分别如何调用？ | 无 | [Worker / HealthStatus / impl Worker](../../../../src/worker.rs) |
| [元组、元组结构体与 unit](tuples-and-unit.md) | () 与 (StatusCode, String) 携带多少信息？ | 无 | [type HttpResponse](../../../../src/gateway/retry.rs) |
| [表达式、返回值与分号](expressions-and-semicolons.md) | 为什么 match 后加分号可能把结果变成 ()？ | 无 | [selected_idex](../../../../src/gateway.rs) |
| [String 与 str：拥有文本还是借用文本](string-and-str.md) | &String 和 &str 是同一种类型吗？ | 无 | [Worker::id](../../../../src/worker.rs) |
| [数组、切片、Vec 与 usize](arrays-slices-and-vec.md) | 当前函数接收 &[Arc<Worker>]，为什么不是 Vec？ | 无 | [healthy_worker_indices](../../../../src/policy.rs) |
| [迭代时借用还是移动](iteration-and-collection.md) | values().map(Arc::clone) 复制了 Worker 吗？ | 数组、切片、Vec 与 usize、共享借用与可变借用 | [WorkerRegistry::snapshot](../../../../src/worker_registry.rs) |
