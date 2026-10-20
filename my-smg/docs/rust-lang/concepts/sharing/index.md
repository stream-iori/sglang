# 共享拥有与修改

[总入口](../../index.md) · [问题索引](../../questions.md)

按表格顺序阅读；遇到具体问题也可以单独进入。前置指概念依赖，不意味着必须提前实施后续功能。

| 文章 | 解决的问题 | 前置 | 源码入口 |
|---|---|---|---|
| [Box：独占拥有堆上对象](box.md) | Box 拥有权与 dyn 的分发方式能否分开理解？ | 共享借用与可变借用、Drop 与 RAII | [Box<dyn Policy + Send>](../../../../src/main.rs) |
| [Rc 与 Arc：多个句柄共同拥有数据](rc-and-arc.md) | Arc::clone 是否创建新的 Worker？ | 共享借用与可变借用、Drop 与 RAII | [WorkerRegistry::get](../../../../src/worker_registry.rs) |
| [Weak 与强引用环](weak-and-cycles.md) | 弱引用还在时数据一定在吗？ | 共享借用与可变借用、Drop 与 RAII | [workers: HashMap<String, Arc<Worker>>](../../../../src/worker_registry.rs) |
| [RefCell：单线程里的运行时借用检查](refcell.md) | 只在单线程运行就能随意同时存在多个 &mut 吗？ | 共享借用与可变借用、Drop 与 RAII | [failures_remaining](../../../../src/bin/fake-worker.rs) |
| [Mutex 与 Guard：锁何时释放](mutex-and-guards.md) | 额外的大括号如何限制持锁时间？ | 共享借用与可变借用、Drop 与 RAII | [state.registry.lock()](../../../../src/gateway.rs) |
| [毒锁：知道状态可能不完整](poisoning.md) | into_inner 与 clear_poison 是否等同于恢复正确状态？ | 共享借用与可变借用、Drop 与 RAII | [CircuitGate::lock_state](../../../../src/circuit_breaker/gate.rs) |
| [Deref：为什么能写 *guard](deref-and-guards.md) | *guard 是取值、复制，还是写入位置？ | 共享借用与可变借用、Drop 与 RAII | [*remaining -= 1](../../../../src/bin/fake-worker.rs) |
| [原子变量与 Relaxed](atomics.md) | fetch_add 返回增加前还是增加后的值？ | 共享借用与可变借用、Drop 与 RAII | [fetch_add / fetch_sub](../../../../src/worker.rs) |
| [Send 与 Sync：哪些值能跨线程](send-and-sync.md) | 为什么 Arc<RefCell<T>> 仍不能直接跨线程共享？ | 共享借用与可变借用、Drop 与 RAII | [Mutex<Box<dyn Policy + Send>>](../../../../src/gateway.rs) |
