# 迭代时借用还是移动

[分类目录](index.md) · [总入口](../../index.md) · [问题索引](../../questions.md)

**iter 借用元素，into_iter 按接收者类型决定如何迭代；collect 需要确定目标容器。**

| 阅读准备 | 内容 |
|---|---|
| 前置 | [数组、切片、Vec 与 usize](arrays-slices-and-vec.md) · [共享借用与可变借用](../ownership/borrowing.md) |
| 源码入口 | [src/worker_registry.rs](../../../../src/worker_registry.rs)：`WorkerRegistry::snapshot` |
| 阅读时机 | 按当前课程需要阅读，不据此跳过 learning 的阶段验收。 |

## 容器迭代时的所有权

三个常用迭代方式分别对应读取、修改和消费：

```rust
let mut names = vec![
    String::from("alice"),
    String::from("bob"),
];

for name in names.iter() {
    println!("{name}");       // name: &String
}

for name in names.iter_mut() {
    name.make_ascii_uppercase(); // name: &mut String
}

for name in names.into_iter() {
    println!("{name}");       // name: String
}

// names 已被 into_iter() 消费，不能再使用
```

| 写法 | 元素类型 | 容器之后能否使用 |
|---|---|---:|
| `iter()` | `&T` | 能 |
| `iter_mut()` | `&mut T` | 能，借用结束后 |
| `into_iter()` | `T` | 不能，容器已被消费 |

`for item in collection` 通常调用 `IntoIterator::into_iter(collection)`，也会消费拥有所有权的
容器；不想消费时写 `for item in &collection`。

## 自查与关联

values().map(Arc::clone) 复制了 Worker 吗？

[struct、enum、impl 与 Self](struct-enum-and-impl.md) · [元组、元组结构体与 unit](tuples-and-unit.md) · [表达式、返回值与分号](expressions-and-semicolons.md)
