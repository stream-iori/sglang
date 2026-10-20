# 从 my-smg 学 Rust

**看不懂语法，去概念短文；不懂库接口，去三方库专题；想读懂完整流程，去项目导读。**

这是已有实现的阅读材料，不代替 [learning 实施课程](../../learning/index.md)。先完成当前阶段的练习和验收，再按需读后续知识；项目代码由学习者亲手实现。

## 选择入口

| 我现在要做什么 | 从这里开始 |
|---|---|
| 从头建立 Rust 基础 | [语言与标准库](concepts/index.md) |
| 看懂 Tokio、Axum、Reqwest、Serde、url | [核心三方库](libraries/index.md) |
| 沿一条实际调用链读源码 | [项目代码导读](walkthroughs/index.md) |
| 查某个符号、报错或常见疑问 | [按问题查找](questions.md) |
| 对照 Java 泛型、上游实现和可选库 | [进阶与扩展](advanced/index.md) |

## 知识前置图

```text
工程与模块 → 类型、值与容器 → 所有权与借用
                                  │
                       ┌──────────┼──────────┐
                       ↓          ↓          ↓
                    模式匹配   错误处理   泛型与 trait
                                             │
                                  ┌──────────┴─────────┐
                                  ↓                    ↓
                               共享状态             Future 基础
                                  └──────────┬─────────┘
                                             ↓
                                     Tokio / HTTP 库
                                             ↓
                                      项目代码导读
```

## 基础阅读与课程对应

| 当前阶段 | 按需阅读 |
|---|---|
| 0：工程起步 | [工程与模块](concepts/project/index.md) |
| 1～2：节点与轮询 | [类型和值](concepts/values/index.md)、[所有权](concepts/ownership/index.md)、[模式](concepts/patterns/index.md) |
| 3：策略抽象 | [类型能力](concepts/types/index.md) |
| 4：配置与错误 | [错误处理](concepts/errors/index.md)、[Serde](libraries/serde/index.md)、[url](libraries/url/index.md) |
| 5～6：HTTP 服务与转发 | [异步机制](concepts/async/index.md)、[Tokio](libraries/tokio/index.md)、[Axum](libraries/axum/index.md)、[Reqwest](libraries/reqwest/index.md) |
| 7～9：并发、管理与重试 | [共享状态](concepts/sharing/index.md)、[项目导读](walkthroughs/index.md) |
| 10 及以后 | 按课程范围理解 [CircuitPermit](walkthroughs/circuit-permit.md)，不把已有熔断代码视为限流、排队或流式阶段已完成 |

## 怎样读一篇

先读结论和前置，再看源码片段与状态图。问自己：类型是什么，数据归谁，作用域何时结束，下一步在什么时候执行？

项目摘录标出源码入口；独立例子和历史阶段接口会单独说明。不要把文档里省略的代码当成可以直接编译的完整程序。

每个知识点只有一个主要讲解位置。其他文章通过“前置”和“关联”连接，不要求你一次读完整个目录。

[文档维护与迁移记录](_maintenance/migration-map.md)
