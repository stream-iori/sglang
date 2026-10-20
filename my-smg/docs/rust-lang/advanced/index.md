# 进阶与扩展

[总入口](../index.md) · [问题索引](../questions.md)

这些主题不作为基础阅读的前置。thiserror/clap 当前未引入；上游材料仅在独立实现和验收之后对照。

| 文章 | 解决的问题 | 前置 | 源码入口 |
|---|---|---|---|
| [协变与不变：从 Java 泛型到 Rust 引用](variance.md) | 外层借用可以缩短，为何不能随意替换内层引用类型？ | Java 泛型与 PECS、生命周期 | [InFlightGuard<'a>](../../../src/worker.rs) |
| [高阶生命周期与函数输入逆变](higher-ranked-lifetimes.md) | fn(&str) 与某个固定 fn(&'a str) 能直接画等号吗？ | 协变与不变 | [Future 约束的阅读前置](../../../src/gateway/retry.rs) |
| [扩展：上游的 Arc 与 Weak 关系](upstream-ownership.md) | 当前 guard 的借用设计与上游共享拥有设计有什么不同？ | Trait、生命周期 | [上游对照入口](../../../learning/04-source-map.md) |
| [扩展：thiserror 的错误派生](thiserror.md) | thiserror 自动生成能力，是否等于自动处理错误？ | Trait、生命周期 | [ConfigLoadError](../../../src/config.rs) |
| [扩展：clap 的参数派生](clap.md) | 派生参数解析和业务校验有什么不同？ | Trait、生命周期 | [env::args](../../../src/bin/fake-worker.rs) |
| [历史案例：生命周期约束检查遗漏](compiler-case-study.md) | 变型规则与编译器检查遗漏为什么是两回事？ | 协变与不变 | [历史实验的上下文](variance.md) |
| [设计对照：责任、机制与策略](design-decisions.md) | 什么情况下底层保留具体错误、高层再决定响应？ | Trait、生命周期 | [加载和校验的责任边界](../../../src/config.rs) |
| [扩展：支付对象的所有权选择](payment-ownership.md) | 只读保存能否使用 &Payment，为什么？ | Trait、生命周期 | [对照拥有值与借用的差异](../../../src/worker.rs) |
| [扩展：Clone 与持久化一致性](payment-consistency.md) | 禁止 Clone 是否能替代乐观锁？ | Trait、生命周期 | [对照 generation 的版本校验](../../../src/circuit_breaker/gate.rs) |
| [Java 泛型与 PECS：变型的入门对照](java-variance.md) | 为什么 List<Cat> 不能作为 List<Animal>？ | Trait、生命周期 | [泛型策略与 Java 对照](../../../src/policy.rs) |
| [两层引用：外层生命周期与内层类型](reference-variance.md) | 为什么缩短借用时间不等于允许写入任意短引用？ | 协变与不变 | [InFlightGuard<'a>](../../../src/worker.rs) |
| [扩展：支付值对象与 Copy](payment-values.md) | 小值对象与可变业务实体应如何区分？ | Trait、生命周期 | [CircuitState 的 Copy 对照](../../../src/circuit_breaker.rs) |
