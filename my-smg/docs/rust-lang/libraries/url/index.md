# url

[总入口](../../index.md) · [问题索引](../../questions.md)

以当前 Cargo.lock 锁定的 **2.5.8** 为例。Manifest 的版本范围和最终锁定版本要分开看。

[对应版本官方文档](https://docs.rs/url/2.5.8/url/)。按表格顺序阅读；遇到具体问题也可以单独进入。前置指概念依赖，不意味着必须提前实施后续功能。

| 文章 | 解决的问题 | 前置 | 源码入口 |
|---|---|---|---|
| [URL 解析与地址校验](parsing-and-validation.md) | 能解析的 URL 是否一定是合法 Worker 地址？ | Option、Result 与容器中的借用和取值 | [GatewayConfig::validate](../../../../src/config.rs) |
