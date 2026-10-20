# Tokio

[总入口](../../index.md) · [问题索引](../../questions.md)

以当前 Cargo.lock 锁定的 **1.53.2** 为例。Manifest 的版本范围和最终锁定版本要分开看。

[对应版本官方文档](https://docs.rs/tokio/1.53.2/tokio/)。按表格顺序阅读；遇到具体问题也可以单独进入。前置指概念依赖，不意味着必须提前实施后续功能。

| 文章 | 解决的问题 | 前置 | 源码入口 |
|---|---|---|---|
| [Runtime、任务、线程与 Mio](runtime-task-thread.md) | Tokio、Mio 与 Netty EventLoop 分别在哪一层？ | Future 与 await | [#[tokio::main]](../../../../src/main.rs) |
| [spawn：启动可独立调度的任务](spawn.md) | 调用 spawn 后是否需要先 await JoinHandle 才开始运行？ | Future 与 await | [start_health_checks](../../../../src/gateway.rs) |
| [JoinHandle：等任务完成，不是任务本体](join-handle.md) | 丢弃 JoinHandle 是否等于取消任务？ | Future 与 await | [handle.await](../../../../src/gateway/tests/support.rs) |
| [sleep 与 timeout：等待和时间上限](time.md) | 超时能证明对端没有执行请求吗？ | Future 与 await | [tokio::time::timeout](../../../../src/gateway.rs) |
| [取消：abort 之后为什么还要 await](cancellation.md) | abort 返回时，guard 一定已经销毁了吗？ | Future 与 await | [task.cancel_and_join().await](../../../../src/gateway/tests/lifecycle.rs) |
| [异步网络：bind、accept 与分段读写](networking.md) | 为什么要累积数据直到读到请求头结束标记？ | Future 与 await | [start_slow_body_server](../../../../src/gateway/tests/support.rs) |
