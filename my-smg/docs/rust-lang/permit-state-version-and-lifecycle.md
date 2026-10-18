# 设计感悟：用 Permit 连接请求、状态版本与生命周期

[返回索引](index.md) · [设计原则](desgin-principle.md) · [其他学习感悟](realization.md)

## 核心感悟

**之前允许开始，不等于现在仍允许提交结果。**

一个异步操作开始后，共享状态可能已经变化。需要分开判断：

| 时刻 | 要回答的问题 | 当前熔断例子的机制 |
|---|---|---|
| 开始时 | 现在能执行吗 | Gate 加锁检查熔断状态，并发放 Permit |
| 完成时 | 这个结果现在还有效吗 | 比较 Permit 保存的版本与 Gate 当前版本 |
| 取消时 | 未完成的机会由谁收尾 | Permit 的 Drop 处理 |

这不是 Rust 专有思想。Rust 的所有权、消费 self 和 Drop，让其中一些约束更容易落实。

<a id="request-state-relationship"></a>

## 1. Permit 显式保存请求与熔断器的关系

CircuitPermit 表达的契约是：

> 这一次尝试，被这个熔断器、在这个状态版本下允许执行。

```text
共享 CircuitGate
    └─ Mutex<GateState>
          ├─ CircuitBreaker：共享状态机
          └─ generation：当前状态版本的身份标记

请求 A → Permit A
           ├─ gate：Arc 指向上述同一个 CircuitGate
           ├─ generation：申请时保存的版本标记
           └─ finished：是否已经报告结果

请求 B → Permit B → 同一个 Gate，不是另一份 Breaker
```

| 对象 | 生命周期与职责 |
|---|---|
| CircuitBreaker | 保存状态，决定状态转换 |
| GateState | 把 breaker 和当前版本放在一起，维护一致性 |
| CircuitGate | 提供共享入口，在锁内检查和更新 |
| CircuitPermit | 每次获准尝试一份，关联共享 Gate 和申请时版本 |

计划集成时，每个 Worker 一套共享 Gate；每次重试重新申请凭证。
这是后续集成方式，不代表当前 HTTP 链路已经使用熔断器。

<a id="state-version-example"></a>

## 2. 为什么需要状态版本：A、B、C 的例子

假设一次失败触发熔断。①②③只是讲解标签，Arc 标记里没有保存这些数字。

| 顺序 | 事件 | 共享状态与版本 | 凭证 |
|---|---|---|---|
| 1 | A、B 在正常状态下获准 | Closed，版本① | A=①，B=① |
| 2 | B 失败，触发熔断 | Open，换成② | A 仍持有① |
| 3 | 冷却后放行探测 C | HalfOpen，换成③ | C=③ |
| 4 | A 的旧成功结果返回 | 仍是 HalfOpen，③ | ①不匹配，忽略 |
| 5 | C 成功返回 | 校验③通过，恢复 Closed 并换版本 | 接受本轮探测结果 |

如果只判断“当前 HalfOpen，收到成功”，就可能把 A 当成探测 C。

```text
请求身份：这是 A、B 还是 C？
状态版本：这个请求在什么状态版本下获准？
```

A、B 共用版本，不是因为同时到达，而是它们申请之间没有状态转换。
之前讨论中的“轮次”容易被误解为请求批次，**状态版本**是更清楚的说法。
当前设计在状态转换时使旧凭证失效，这是一个明确的业务策略，不是所有系统都必须采用的规则。

## 3. Arc<()> 保存什么

它不保存请求 ID、状态值或版本数字。`()` 没有业务数据，Arc 用来共享一个可比较身份的标记。

| 操作 | 含义 |
|---|---|
| `Arc::new(())` | 创建一个新身份标记 |
| `Arc::clone(&marker)` | 共同持有同一个标记，不创建新身份 |
| `Arc::ptr_eq(&a, &b)` | 检查是否同一个标记 |

Java 中可以类比：

```java
Object currentVersion = new Object();
Object requestVersion = currentVersion;

currentVersion = new Object();

boolean stillValid = requestVersion == currentVersion; // false
```

也可以改用数字版本，但必须避免版本溢出后重复使用而误认旧凭证。
必要的是“识别过期结果”，不是 generation 这个名字，也不是 Arc<()> 这一种实现。
详见 [Arc：对象身份与值相等](arc-weak-and-cycle-references.md#ptr_eq对象身份与值相等)。

<a id="design-principles"></a>

## 4. 与 SOLID 和系统设计原则的关系

| 原理 | 当前例子 | 边界 |
|---|---|---|
| 关注点分离、单一职责 | Breaker 管转换，Gate 管同步，Permit 管单次操作生命周期 | 是职责边界，不是类越多越好 |
| 关系显式建模 | Permit 保存哪个 Gate、哪个状态版本 | 不依赖调用方临时拼接全局状态 |
| 版本校验 | 结果提交前判断版本是否有效 | Rust 类型系统不会自动判断业务结果是否过期 |
| 所有权跟随责任 | Permit 持有 Gate 的 Arc，消费 self 报告结果 | Arc 保证对象存活，不保证结果仍有效 |
| 生命周期绑定清理 | 未完成的凭证 Drop 时处理取消 | 不等于取消远端已经执行的业务副作用 |
| 封装不变量 | 检查、占用探测机会、更新版本在同一把锁内进行 | 锁防止交错，不会自动修复 panic 中断的更新 |

直接对应 SOLID 的主要是 S（单一职责）。版本控制和 RAII 是其他设计思想，不必强行归入 SOLID 的每一项。
本例也不是完整的能力安全体系，只是用对象显式表达一次操作的授权与完成责任。

### 编译器能帮什么，不能帮什么

```text
success(self) / failure(self) + 不实现 Clone
    → 同一个凭证不能被普通代码重复提交

#[must_use]
    → 提醒调用方不要忽略凭证，不是强制保持正确作用域

Drop
    → 正常作用域退出、展开式 panic、任务取消并销毁 Future 时清理

运行时版本比较
    → 判断这次结果是否仍属于当前状态版本
```

过早 drop 仍可能导致业务统计错误；进程直接终止、abort 或刻意遗忘对象时也不能保证执行 Drop。
这是“让正确使用更容易、让错误使用更难”，不是“所有错误都被编译器消灭”。

## 5. 可以迁移到哪些场景

```text
申请权限 → 得到凭证 → 执行操作 → 提交结果前校验 → 完成或取消清理
```

| 场景 | 类似的问题 |
|---|---|
| 页面搜索 | 旧查询返回，覆盖新查询结果 |
| 缓存刷新 | 旧刷新任务覆盖更新的数据 |
| 数据库乐观锁 | 基于旧版本的写入覆盖新版本 |
| 主节点切换 | 旧主仍试图提交操作 |

这些只是问题结构的类比，不能直接复用同一套实现。
当前 Arc 标记只在本进程内做身份比较，不等于分布式 fencing token。
分布式场景需要执行写入的一方真正拒绝旧令牌；本地忽略返回结果不能撤销远端副作用。

## 6. 当前练习还缺什么

- HTTP 集成：何时申请、每次重试如何记录结果、健康检查如何与熔断区分。
- 毒锁策略：当前 into_inner 只拿回守卫，不证明 breaker 与版本依然一致。
- 取消验收：除直接 drop 外，还需要覆盖实际异步任务取消。

因此不要把这份设计感悟当作“并发集成已经完成”的验收记录。
毒锁问题见 [MutexGuard 与毒锁](mutex-guard-deref-and-poisoning.md)，
取消过程见 [Tokio：abort 与 Drop](tokio-basics-and-task-cancellation.md#7-abort-与-drop取消测试完整过程)。

## 一句话总结

**好的抽象不只是减少重复代码，而是把授权、有效性和收尾责任变成明确的契约。**

继续阅读：[所有权与借用](ownership-move-and-borrowing.md)、
[设计原则：所有权跟随责任与机制/策略分离](desgin-principle.md)。
