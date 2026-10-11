# 类型别名与关联类型：HttpResponse 和 Future::Output

[返回索引](index.md)

| 写法 | 概念 | 大白话 |
|---|---|---|
| `type HttpResponse = (StatusCode, String);` | 类型别名 | 给已有类型取另一个名字 |
| `Future::Output` | trait 的关联类型 | 某种 Future 完成后的结果类型 |
| `Future<Output = T>` | 关联类型等式约束 | 要求这个 Future 完成后得到 T |
| `Result<T, E>` | 泛型类型参数 | 为 Result 指定成功和错误类型 |

关键纠正：`Output` 不是 Future 的普通泛型参数。

<a id="type-alias"></a>

## 1. type HttpResponse：别名，不是新类型

当前 `src/gateway/retry.rs` 有：

```rust
type HttpResponse = (StatusCode, String);
```

```text
HttpResponse ──另一个名字──> (StatusCode, String)
```

它不创建变量、不分配内存、不包装元组，也不改变所有权。
这里的名字不是 Axum 的 Response，也不是 Reqwest 的 Response，只是我们给元组起的名字。

```rust
let response: HttpResponse = (StatusCode::OK, "hello".to_string());
let tuple: (StatusCode, String) = response;
```

不需要 `.into()`：二者本来就是同一个类型。上面赋值仍会移动含 String 的元组，
别名不会让它突然具备 Copy 能力。

| 写法 | 是否建立独立类型 | 是否有新包装层 |
|---|---|---|
| `type HttpResponse = (StatusCode, String);` | 否 | 否 |
| `struct HttpResponse(StatusCode, String);` | 是 | 有结构体边界 |

结构体需要通过构造表达式包装字段；别名没有独立构造器。
若要让编译器区分两种业务值，应该考虑新类型，而不是仅换别名。

## 2. 同一个别名可以出现在 Ok 和 Err 中

```rust
Result<HttpResponse, HttpResponse>
```

等同于：

```rust
Result<(StatusCode, String), (StatusCode, String)>
```

| 分支 | 当前重试练习中的含义 |
|---|---|
| `Ok(response)` | 收到完整上游响应，允许继续判断状态码 |
| `Err(response)` | 发送或读取失败，已经转换成网关错误响应，直接结束 |

`Ok` 不代表状态码一定是 200：真实上游返回 503 仍可放在 Ok 中。
具体类型相同，不等于分支语义相同。

最初使用泛型 E 时，E 也不是 Error 的缩写约束：除非声明 `E: Error`，
否则它可以是任意满足当前接口的类型。本例 Err 内的值是响应元组，
编译器因此可以推断 E 为该元组，而不是原始的 reqwest::Error。

错误控制流见 [Result、? 与错误传播](result-question-mark-and-error-propagation.md)。

<a id="associated-output"></a>

## 3. Output：Future 的关联类型

Future trait 内声明了 `type Output;`。只看这个声明的概念骨架：

```rust
trait Future {
    type Output;
    // 省略 poll 等定义；这不是完整的标准库源码。
}
```

实现 Future 的具体类型，需要确定自己的 Output。
普通 async 函数和异步块由编译器生成对应的 Future 实现，不需要我们手写。

```rust
async fn answer() -> usize {
    42
}
```

```text
answer()
    ↓ 返回一个编译器生成的 Future 类型
该类型实现 Future，Output = usize
    ↓ .await
42：usize
```

对于直接实现 Future 的类型，`.await` 得到的就是它的 Output。
更一般地，可等待的值也可以先通过 IntoFuture 转成 Future，见 Tokio 专题。

<a id="generic-vs-associated"></a>

## 4. 为什么写 Output =，而不是 Future<usize>？

| 语法 | 给谁指定类型 |
|---|---|
| `Result<usize, String>` | Result 声明的两个泛型参数 |
| `Future<Output = usize>` | Future 声明的关联类型 Output |

Future 本身没有声明一个普通类型参数供 `Future<usize>` 使用。
尖括号不只用于传泛型参数，也可以写关联类型约束，`Output =` 的等号不能省略。

对于给定的具体 Future 实现，Output 已由实现确定。调用者的等式约束是在检查它是否符合要求，
不是让同一个 Future 在调用时随意选择不同的完成结果。

```text
泛型参数：接口上可传入的类型槽位
关联类型：某个 trait 实现确定的类型
```

## 5. 拆解当前重试函数的约束

```rust
F: FnMut() -> Fut,
Fut: Future<Output = Result<HttpResponse, HttpResponse>>,
```

| 名称 | 是什么 |
|---|---|
| F | 闭包的具体类型，函数的泛型参数 |
| Fut | 闭包调用后产生的 Future 类型，函数的泛型参数 |
| Output | Future trait 中定义的关联类型名称，不是额外声明的泛型参数 |
| HttpResponse | 元组的类型别名 |

```text
operation：F
    ↓ operation()：调用闭包
Future：Fut
    ↓ .await：取得 Output
Result<HttpResponse, HttpResponse>
    ↓ ?
Ok：取出元组继续判断状态；Err：从重试函数提前返回
```

注意：这里的 Output 是整个 Result，不只是其中的 HttpResponse。
闭包是可重复调用的创建办法，每次创建新的 Future；完成过的 Future 不是可重放的任务模板。
关联阅读：[闭包与 Fn 系列](closures-and-fn-traits.md)。

## 6. 放进 timeout 后为什么有两层 Result？

```text
重试 Future 的 Output：Result<HttpResponse, HttpResponse>
                  ↓ timeout 包装，再 .await
Result<Result<HttpResponse, HttpResponse>, Elapsed>
```

| 匹配 | 含义 |
|---|---|
| `Ok(Ok(response))` | 按时完成，收到完整响应 |
| `Ok(Err(response))` | 按时完成，但发送或读取失败 |
| `Err(elapsed)` | 总期限到达 |

外层 Ok 只表示按时完成，不表示里面的操作成功。
见 [Tokio：sleep、timeout 和 pending](tokio-basics-and-task-cancellation.md#6-sleeptimeout-和-pending)。

## 关联阅读与依据

- [Trait、泛型与分发](traits-dispatch-and-smart-pointers.md)。
- [Axum：Handler 的 Future 约束](axum-routing-handler-and-future.md)：其中 `Future<Output = Res>` 是同一种关联类型约束。
- [Rust Reference：类型别名](https://doc.rust-lang.org/reference/items/type-aliases.html)。
- [Rust Reference：关联类型](https://doc.rust-lang.org/reference/items/associated-items.html#associated-types)。
- [标准库 Future](https://doc.rust-lang.org/std/future/trait.Future.html)。
