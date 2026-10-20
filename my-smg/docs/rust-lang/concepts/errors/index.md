# 错误、转换与传播

[总入口](../../index.md) · [问题索引](../../questions.md)

按表格顺序阅读；遇到具体问题也可以单独进入。前置指概念依赖，不意味着必须提前实施后续功能。

| 文章 | 解决的问题 | 前置 | 源码入口 |
|---|---|---|---|
| [Option、Result 与容器中的借用和取值](option-and-result.md) | as_ref、as_deref 和 take 分别留下什么？ | struct、enum、impl 与 Self、共享借用与可变借用 | [handle: Option<JoinHandle<T>>](../../../../src/gateway/tests/support.rs) |
| [? 与 ??：成功继续，失败提前返回](question-mark.md) | Result 与 Option 为什么不能直接混用问号？ | Option、Result 与容器中的借用和取值 | [operation().await?](../../../../src/gateway/retry.rs) |
| [From 与 Into：为什么有两个转换入口](from-and-into.md) | into 是强制转换、clone，还是提前返回？ | Trait | [impl From<std::io::Error> for ConfigLoadError](../../../../src/config.rs) |
| [TryFrom、TryInto 与可能失败的转换](fallible-conversions.md) | parse 和 try_into 分别转换什么来源？ | if let 与 let else | [parse::<u64>](../../../../src/bin/fake-worker.rs) |
| [错误枚举、Error 与 source](error-and-source.md) | From 会拿走错误，source 会拿走吗？ | if let 与 let else | [ConfigLoadError / Error::source](../../../../src/config.rs) |
| [Debug、Display 与 Formatter](debug-display-and-formatter.md) | 为什么 fmt 接收 &mut Formatter<'_>？ | if let 与 let else | [impl Display for ConfigLoadError](../../../../src/config.rs) |
| [unwrap、expect 与可恢复失败](unwrap-and-expect.md) | 正式配置加载为什么用 Result 而不是 expect？ | if let 与 let else | [expect("managed task should own its handle")](../../../../src/gateway/tests/support.rs) |
| [??：逐层拆开 Result 与 Option](nested-results.md) | Ok(Err(error)) 与 Err(error) 分别在哪一步返回？ | ? 与 ?? | [timeout 的两层结果](../../../../src/gateway.rs) |
| [转换与所有权：into 到底消费什么](conversion-and-ownership.md) | 为什么 String.into 和 str 引用.into 对原值的影响不同？ | if let 与 let else | [error.into()](../../../../src/main.rs) |
