# 项目代码导读

[总入口](../index.md) · [问题索引](../questions.md)

按表格顺序阅读；遇到具体问题也可以单独进入。前置指概念依赖，不意味着必须提前实施后续功能。

| 文章 | 解决的问题 | 前置 | 源码入口 |
|---|---|---|---|
| [代码导读：Worker 与在途请求的生命周期](worker-lifecycle.md) | 请求被取消时，谁负责计数减一？ | 共享借用与可变借用、? 与 ?? | [Worker / InFlightGuard](../../../src/worker.rs) |
| [代码导读：轮询选择与策略状态](policy-selection.md) | 每次新建 RoundRobin 为什么总从第一个开始？ | 共享借用与可变借用、? 与 ?? | [RoundRobin::select](../../../src/policy.rs) |
| [代码导读：配置加载与分层错误](configuration-loading.md) | 哪一层决定错误怎样打印，哪一层保留底层原因？ | 共享借用与可变借用、? 与 ?? | [GatewayConfig::from_file](../../../src/config.rs) |
| [代码导读：一次聊天请求](chat-forwarding.md) | 为什么取得快照后才能释放注册表锁并等待网络？ | State、Client 与请求构建、Mutex 与 Guard | [chat](../../../src/gateway.rs) |
| [代码导读：重试与总期限](retry-and-timeout.md) | Err(response) 与上游 503 是否走相同的重试分支？ | 关联类型、闭包与 Fn、FnMut、FnOnce、sleep 与 timeout | [retry_with_backoff](../../../src/gateway/retry.rs) |
| [代码导读：ManagedTask 的清理责任](task-cleanup.md) | Drop 的兜底和显式 cancel_and_join 分别保证什么？ | self、mut self、&self 与 &mut self、Option、Result 与容器中的借用和取值、取消 | [ManagedTask / TestServer](../../../src/gateway/tests/support.rs) |
| [代码导读：许可、状态版本与迟到结果](circuit-permit.md) | 为什么 Arc<()> 身份比只比较状态枚举更有用？ | 共享借用与可变借用、? 与 ?? | [CircuitPermit](../../../src/circuit_breaker/gate.rs) |
