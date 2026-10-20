# 工程与模块

[总入口](../../index.md) · [问题索引](../../questions.md)

按表格顺序阅读；遇到具体问题也可以单独进入。前置指概念依赖，不意味着必须提前实施后续功能。

| 文章 | 解决的问题 | 前置 | 源码入口 |
|---|---|---|---|
| [Package 与 Crate：一个工程里有几棵编译树](package-and-crate.md) | 为什么 main.rs 不能访问库的 pub(crate) 项？ | 无 | [[package]](../../../../Cargo.toml) |
| [模块声明如何找到文件](modules-and-files.md) | circuit_breaker.rs 与 circuit_breaker/ 是两个模块吗？ | 无 | [pub mod gate](../../../../src/circuit_breaker.rs) |
| [crate、self、super 与 use](paths-and-imports.md) | super 指父类还是父模块？ | 无 | [use super::is_retryable_status](../../../../src/gateway/retry.rs) |
| [pub 与受限可见性](visibility.md) | 祖父模块能否访问 pub(super) 的函数？ | 无 | [pub(super) struct GatewayState](../../../../src/gateway.rs) |
| [pub use：内部组织与公开入口](reexports.md) | 公开入口变短是否意味着目录变浅？ | 无 | [pub mod worker](../../../../src/lib.rs) |
| [edition：语言规则与编译器版本](edition.md) | edition、rust-version 与 package version 分别管什么？ | 无 | [edition = "2024"](../../../../Cargo.toml) |
| [Workspace：统一管理多个包](workspace.md) | 当前 my-smg 是 workspace 成员集合还是一个 package？ | 无 | [[package]](../../../../Cargo.toml) |
| [Workspace 的继承、依赖与构建选择](workspace-configuration.md) | workspace.dependencies 是否自动添加依赖？ | 无 | [[dependencies]](../../../../Cargo.toml) |
| [Workspace 命令：选择构建哪些成员](workspace-commands.md) | -p 接受目录名还是 package.name？ | Workspace | [package 名与构建目标](../../../../Cargo.toml) |
