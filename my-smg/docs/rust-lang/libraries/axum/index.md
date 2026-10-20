# Axum

[总入口](../../index.md) · [问题索引](../../questions.md)

以当前 Cargo.lock 锁定的 **0.8.9** 为例。Manifest 的版本范围和最终锁定版本要分开看。

[对应版本官方文档](https://docs.rs/axum/0.8.9/axum/)。按表格顺序阅读；遇到具体问题也可以单独进入。前置指概念依赖，不意味着必须提前实施后续功能。

| 文章 | 解决的问题 | 前置 | 源码入口 |
|---|---|---|---|
| [Router 与 Handler：注册和执行是两件事](router-and-handler.md) | 注册路由时 chat 已执行了吗？ | 解构、Future 与 await | [.route("/chat", post(chat))](../../../../src/main.rs) |
| [State：框架怎样准备共享状态](state.md) | 框架根据参数名字还是参数类型选择提取规则？ | 解构、Future 与 await | [State(state): State<Arc<GatewayState>>](../../../../src/gateway.rs) |
| [用最小模型理解 Extractor](extractor-mechanics.md) | 只定义 State 结构体是否就具备提取能力？ | 泛型、类型推断与 turbofish、Trait、解构 | [chat](../../../../src/gateway.rs) |
| [Json 与 Path：从请求准备参数](json-and-path.md) | State、Json、Path 分别从哪里取数据？ | 解构、Future 与 await | [add_worker / remove_worker](../../../../src/gateway.rs) |
| [响应转换：handler 可以返回什么](responses.md) | 返回 Result 和返回 HTTP 500 是同一层概念吗？ | 解构、Future 与 await | [chat / health](../../../../src/gateway.rs) |
