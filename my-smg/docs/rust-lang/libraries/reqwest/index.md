# Reqwest

[总入口](../../index.md) · [问题索引](../../questions.md)

以当前 Cargo.lock 锁定的 **0.13.5** 为例。Manifest 的版本范围和最终锁定版本要分开看。

[对应版本官方文档](https://docs.rs/reqwest/0.13.5/reqwest/)。按表格顺序阅读；遇到具体问题也可以单独进入。前置指概念依赖，不意味着必须提前实施后续功能。

| 文章 | 解决的问题 | 前置 | 源码入口 |
|---|---|---|---|
| [Client 与请求构建](client-and-request.md) | Axum post 和 Reqwest post 分别做什么？ | Option、Result 与容器中的借用和取值、Future 与 await | [state.client / client.post(url_ref)](../../../../src/gateway.rs) |
| [响应头、正文与错误分类](response-and-errors.md) | 收到 503 为什么仍可能是 Ok(response)？ | Option、Result 与容器中的借用和取值、Future 与 await | [upstream.status() / upstream.text().await](../../../../src/gateway.rs) |
| [单次请求超时与总超时](timeouts.md) | 1 秒与 2 秒这两层超时各覆盖什么？ | Option、Result 与容器中的借用和取值、Future 与 await | [.timeout(Duration::from_secs(1))](../../../../src/gateway.rs) |
