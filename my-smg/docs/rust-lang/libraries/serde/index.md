# Serde 与 serde_json

[总入口](../../index.md) · [问题索引](../../questions.md)

以当前 Cargo.lock 锁定的 **serde 1.0.229 / serde_json 1.0.154** 为例。Manifest 的版本范围和最终锁定版本要分开看。

[对应版本官方文档](https://docs.rs/serde/1.0.229/serde/)。按表格顺序阅读；遇到具体问题也可以单独进入。前置指概念依赖，不意味着必须提前实施后续功能。

| 文章 | 解决的问题 | 前置 | 源码入口 |
|---|---|---|---|
| [Serialize 与 Deserialize](serialize-and-deserialize.md) | 结构体字段公开与能否序列化是否是同一回事？ | struct、enum、impl 与 Self、derive 与 Default | [#[derive(Deserialize, Serialize)]](../../../../src/gateway.rs) |
| [字段与枚举的 JSON 映射](field-and-enum-mapping.md) | round-robin 怎样对应 RoundRobin？ | struct、enum、impl 与 Self、derive 与 Default | [#[serde(rename_all = "kebab-case")]](../../../../src/config.rs) |
| [JSON 解析与业务校验](json-parsing-and-validation.md) | 空 workers 能被解析，为什么仍然不能启动？ | struct、enum、impl 与 Self、derive 与 Default | [from_json_validated](../../../../src/config.rs) |
