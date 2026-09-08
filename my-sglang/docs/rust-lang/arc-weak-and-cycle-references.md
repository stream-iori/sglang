# Rust：`Arc`、`Weak` 与循环引用

## 结论

`Weak<T>` 用于“能找到对象，但不拥有对象”。它不增加强引用计数，因此把循环中的至少一条边改成
`Weak` 后，对象就能正常释放。

```rust
let weak_context = Arc::downgrade(&app_context);
```

这句把：

```text
app_context: Arc<AppContext>
        │
        │ 借用，不转移所有权
        ▼
Arc::downgrade(&app_context)
        │
        ▼
weak_context: Weak<AppContext>
```

## `Arc` 与 `Weak` 的区别

| 类型 | 是否拥有对象 | 是否增加强引用计数 | 能否保证对象存活 |
|---|---:|---:|---:|
| `Arc<T>` | 是 | 是 | 是 |
| `Weak<T>` | 否 | 否 | 否 |

可以把强引用计数理解成对象的“存活票数”：

```text
strong_count > 0  -> T 一定还活着
strong_count == 0 -> T 被销毁
```

`Weak<T>` 不会让 `strong_count` 增加，所以它不能单独让 `T` 继续存活。

## 为什么两个 `Arc` 会泄漏

假设父子双方都用 `Arc` 保存对方：

```text
外部 ──Arc──> Parent ──Arc──> Child
                  ▲                 │
                  └─────Arc─────────┘
```

当外部引用释放后：

```text
Parent strong_count = 1  （Child 持有）
Child  strong_count = 1  （Parent 持有）
```

两个对象都在等待对方先释放，但谁也不会先释放：

```text
Parent 释放条件：strong_count == 0  ✗
Child  释放条件：strong_count == 0  ✗
```

结果是内存泄漏。

## 用 `Weak` 断开“反向指针”

通常让“拥有关系”保持 `Arc`，让“回指 / 父指针 / 注册表反查”变成 `Weak`：

```text
外部 ──Arc──> Parent ──Arc──> Child
                  ▲                 │
                  └────Weak─────────┘
```

外部释放时，释放顺序变成：

```text
1. 外部 Arc<Parent> 被 drop
2. Parent strong_count = 0 -> Parent 被释放
3. Parent 持有的 Arc<Child> 被 drop
4. Child strong_count = 0 -> Child 被释放
5. Child 中的 Weak<Parent> 自动失效
```

关键不是“`Weak` 自动清理循环”，而是：

```text
Weak 不算拥有者
=> 它不会阻止 strong_count 降到 0
=> 对象可以析构
```

## `upgrade()`：弱引用不能直接使用

`Weak<T>` 指向的对象可能已经释放。使用前必须升级：

```rust
if let Some(context) = weak_context.upgrade() {
    // context: Arc<AppContext>
    // AppContext 仍存活；在此作用域内可安全使用。
} else {
    // 没有任何 Arc<AppContext> 了，对象已经释放。
}
```

类型变化：

```text
Weak<AppContext>
       │ .upgrade()
       ▼
Option<Arc<AppContext>>
       │
       ├─ Some(Arc<_>)：对象存活
       └─ None        ：对象已释放
```

不能把 `Weak` 当作永远有效的 `Arc`：

```rust
// ❌ weak_context.do_something()

// ✅ 先升级
let context = weak_context.upgrade()?;
context.do_something();
```

## SGL Model Gateway 中的例子

SMG 启动时创建 `AppContext` 后，创建 JobQueue：

```rust
let app_context = Arc::new(...);
let weak_context = Arc::downgrade(&app_context);
let worker_job_queue = JobQueue::new(JobQueueConfig::default(), weak_context);
```

源码见 [`server.rs:810-823`](../../../sgl-model-gateway/src/server.rs#L810-L823)。关系是：

```text
AppContext ──Arc──> JobQueue
     ▲                  │
     └──────Weak────────┘
```

如果 JobQueue 反过来持有 `Arc<AppContext>`，就会形成强引用环：

```text
AppContext ──Arc──> JobQueue
     ▲                  │
     └──────Arc─────────┘  <- 无法释放
```

用 `Weak<AppContext>` 后，关闭服务并释放最后一个 `Arc<AppContext>` 时，JobQueue 不会阻止
AppContext 析构。

## 最小可运行模型

```rust
use std::sync::Arc;

let strong = Arc::new(String::from("hello"));
let weak = Arc::downgrade(&strong);

assert_eq!(Arc::strong_count(&strong), 1);
assert_eq!(Arc::weak_count(&strong), 1);

assert_eq!(weak.upgrade().as_deref(), Some("hello"));

drop(strong);

assert!(weak.upgrade().is_none());
```

`drop(strong)` 后，字符串已被释放；变量 `weak` 仍然存在，但它只能返回 `None`。

## 判断该用哪个

| 关系 | 建议 | 原因 |
|---|---|---|
| 父拥有子 | `Arc<Child>` | 父存在时子必须存在 |
| 子回指父 | `Weak<Parent>` | 子不应该延长父的生命周期 |
| 缓存 / registry 中的可失效条目 | 常用 `Weak<T>` | 缓存不应成为对象唯一所有者 |
| 独立且必须共同存活的共享对象 | `Arc<T>` | 调用方需要明确拥有它 |

## 一句话记忆

```text
Arc：我拥有你，你不能先消失。
Weak：我认识你；你还在我才用你。
```
