# match 分支与守卫的执行顺序

[分类目录](index.md) · [总入口](../../index.md) · [问题索引](../../questions.md)

**从上往下匹配，只执行第一个模式和守卫都成功的分支。**

| 阅读准备 | 内容 |
|---|---|
| 前置 | [struct、enum、impl 与 Self](../values/struct-enum-and-impl.md) |
| 源码入口 | [src/gateway/tests/support.rs](../../../../src/gateway/tests/support.rs)：`TestServer::stop` |
| 阅读时机 | 按当前课程需要阅读，不据此跳过 learning 的阶段验收。 |

## match 分支守卫：模式后面还可以加条件

`my-smg` 的测试服务器清理函数有这一段：

```rust
match self.task.cancel_and_join().await {
    Ok(()) => Ok(()),
    Err(error) if error.is_cancelled() => Ok(()),
    Err(error) => Err(error),
}
```

`Err(error) if error.is_cancelled()` 分两步：先匹配 Err 并绑定 error，再检查布尔条件。
条件不满足时，继续尝试后面的分支；不是直接退出 match。

| 输入 | 选中的分支 | 清理结果 |
|---|---|---|
| 正常完成 | `Ok(())` | 成功 |
| 因取消结束 | 带 if 的 Err 分支 | 成功 |
| panic 等其他任务错误 | 最后一个 Err 分支 | 保留错误 |

普通模式负责匹配结构，分支守卫负责附加条件。守卫可以为 false，因此不能只靠一个带守卫的
分支覆盖所有 Err；这里保留无条件的最后一个分支。
参见 [Rust Reference：match guards](https://doc.rust-lang.org/reference/expressions/match-expr.html#match-guards)。

先用整数看分支顺序：

```rust
let value = Some(-2);
let label = match value {
    Some(number) if number > 0 => "positive",
    Some(_) => "non-positive",
    None => "missing",
};
assert_eq!(label, "non-positive");
```

```text
Some(-2)
   ↓ 第一个模式匹配成功：number = -2
number > 0 为 false
   ↓ 继续尝试第二个分支
Some(_) 匹配成功 → 返回 non-positive
```

这段逻辑适用于“服务器清理允许已完成”的场景。取消行为测试仍需要明确检查
`JoinError::is_cancelled()`，不能用“清理成功”替代“确实被取消”的断言。见
[Tokio 资源管理](../../walkthroughs/task-cleanup.md)。

## 自查与关联

两个 Err(error) 为什么不会都执行？

[if let 与 let else：成功值在哪里可用](if-let-and-let-else.md) · [解构：从 State、结构体和元组中取值](destructuring.md) · [ref、引用模式与解引用](reference-patterns.md)
