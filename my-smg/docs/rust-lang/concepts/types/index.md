# 泛型与类型能力

[总入口](../../index.md) · [问题索引](../../questions.md)

按表格顺序阅读；遇到具体问题也可以单独进入。前置指概念依赖，不意味着必须提前实施后续功能。

| 文章 | 解决的问题 | 前置 | 源码入口 |
|---|---|---|---|
| [泛型、类型推断与 turbofish](generics-and-inference.md) | pending::<()>() 中两个 () 各做什么？ | 数组、切片、Vec 与 usize | [ManagedTask<T>](../../../../src/gateway/tests/support.rs) |
| [Trait：类型需要提供什么能力](traits-and-bounds.md) | 为什么实现了 Policy 之后还可能需要把 trait 引入作用域？ | 共享借用与可变借用 | [Policy / impl Policy for RoundRobin](../../../../src/policy.rs) |
| [关联类型：实现确定自己的 Output](associated-types.md) | Output = u32 是转换结果还是检查已有实现？ | 泛型、类型推断与 turbofish、Trait | [Fut: Future<Output = Result<HttpResponse, HttpResponse>>](../../../../src/gateway/retry.rs) |
| [类型别名与新类型](aliases-and-newtypes.md) | 为什么类型别名不需要 into 就能赋给原类型？ | 共享借用与可变借用 | [HttpResponse](../../../../src/gateway/retry.rs) |
| [静态与动态分发](static-and-dynamic-dispatch.md) | Box<dyn Policy> 中 Box 与 dyn 各负责什么？ | 共享借用与可变借用 | [select_with_policy / select_with_dynamic_policy](../../../../src/policy.rs) |
| [哪些 trait 能作为 dyn Trait](dyn-compatibility.md) | 为什么不是任意 trait 都能写成 dyn Trait？ | 共享借用与可变借用 | [trait Policy](../../../../src/policy.rs) |
| [闭包与 Fn、FnMut、FnOnce](closures-and-fn.md) | FnOnce 是否代表运行时最多只能调用一次？ | 共享借用与可变借用 | [wait_until](../../../../src/gateway/tests/support.rs) |
| [derive 与 Default：编译器替你生成什么](derive-and-default.md) | derive(Default) 与手写 new 是否必然一致？ | struct、enum、impl 与 Self | [#[derive(Default)]](../../../../src/worker_registry.rs) |
| [相等、排序与哈希](comparison-and-hash.md) | 为什么浮点数会阻止某些 Eq 和 Ord 派生？ | 共享借用与可变借用 | [#[derive(Debug, Deserialize, PartialEq, Eq)]](../../../../src/config.rs) |
| [PartialEq 与 Eq：怎样定义相等](equality.md) | 两个 Arc 指向的值相等是否表示同一个对象？ | 共享借用与可变借用 | [HealthStatus](../../../../src/worker.rs) |
| [类型推断、parse、collect 与空 Vec](type-inference.md) | 为什么空 Vec 单独写出来时编译器可能无法知道元素类型？ | 共享借用与可变借用 | [parse::<u64>](../../../../src/bin/fake-worker.rs) |
