# Rust：`&[Worker]` 与 `Vec::<usize>::new()`

## 一句话

```rust
fn healthy_worker_indices(workers: &[Worker]) -> Vec<usize>
```

这段签名表示：函数临时借用一段连续的 `Worker`，并返回一个自己拥有的、元素类型为 `usize`
的动态数组。

```rust
Vec::<usize>::new()
```

表示创建一个元素类型明确为 `usize` 的空 `Vec`。`::<usize>` 用于在没有元素可供推断时告诉
编译器具体类型。

## `&[Worker]` 拆开看

| 部分 | 含义 |
|---|---|
| `Worker` | 单个元素的类型 |
| `[Worker]` | 一段连续 `Worker` 的切片；其长度不写在类型里 |
| `&[Worker]` | 对该切片的共享借用 |

切片本身可以理解为一幅“窗口”：它记录数据的起始位置和长度，但不拥有窗口背后的节点。

```text
workers: Vec<Worker>（拥有节点）
        │
        │ &workers
        ▼
&[Worker]（临时借用整段节点）
```

因此下面的函数不接管节点列表：

```rust
fn healthy_worker_indices(workers: &[Worker]) -> Vec<usize> {
    // 只读取 workers
}
```

调用结束后，原来的 `Vec<Worker>` 仍由调用者拥有，可以继续使用：

```rust
let workers = vec![/* Worker 值 */];
let indices = healthy_worker_indices(&workers);

// workers 仍然有效
println!("worker count: {}", workers.len());
```

## 为什么参数不写成 `Vec<Worker>`

如果按值接收：

```rust
fn healthy_worker_indices(workers: Vec<Worker>) -> Vec<usize>
```

调用时通常会把整个 `Vec` 的所有权移动进函数。调用者随后不能再使用原变量，除非函数把它
返回，或者调用前复制数据。

使用 `&[Worker]` 有两个优点：

1. 函数只借用数据，不取得所有权，也不需要克隆节点。
2. 参数不局限于 `Vec<Worker>`，还可以借用数组或某个局部范围。

```rust
healthy_worker_indices(&workers);       // 借用整个 Vec
healthy_worker_indices(&workers[1..3]); // 借用其中一段
```

`&[Worker]` 是共享借用，所以函数可以读取节点，但不能通过它修改节点。需要修改元素时，参数
类型才会是 `&mut [Worker]`。

## `[Worker]` 与 `[Worker; N]` 不同

```text
[Worker; 3]  长度为 3 的数组，长度属于类型的一部分
[Worker]     长度运行时才知道的切片，通常通过 &[Worker] 使用
Vec<Worker>  拥有堆上元素、长度可以增长的动态数组
```

切片类型 `[Worker]` 的大小在编译期不固定，因此一般不会单独按值使用，而是放在引用后面：
`&[Worker]` 或 `&mut [Worker]`。

## `Vec::<usize>::new()` 拆开看

| 部分 | 含义 |
|---|---|
| `Vec` | 标准库的动态数组类型 |
| `usize` | 这个 `Vec` 保存的元素类型 |
| `::new()` | 调用 `Vec` 的关联函数 `new` 创建空值 |
| `::<usize>` | 为泛型参数指定类型的 turbofish 语法 |

因此：

```rust
let indices = Vec::<usize>::new();
```

等价于把类型写在变量一侧：

```rust
let indices: Vec<usize> = Vec::new();
```

两者都会创建一个长度为 0、元素类型为 `usize` 的 `Vec`。

## 为什么空 `Vec` 经常需要类型提示

下面的值没有元素：

```rust
let values = Vec::new();
```

如果后续代码也没有提供信息，编译器无法判断它究竟是 `Vec<usize>`、`Vec<String>`，还是其他
类型的 `Vec`。这时需要显式标注类型。

有元素或周围上下文足够时，编译器通常能推断：

```rust
let indices = vec![0_usize, 2_usize];

let mut indices = Vec::new();
indices.push(0_usize); // 从 push 的参数推断为 Vec<usize>
```

函数返回类型也能提供上下文：

```rust
fn empty_indices() -> Vec<usize> {
    Vec::new()
}
```

这里不需要 turbofish，因为 `-> Vec<usize>` 已经告诉编译器返回值的具体类型。

## 为什么索引使用 `usize`

Rust 的切片和 `Vec` 使用 `usize` 表示长度与索引：

```rust
let index: usize = 0;
let worker = &workers[index];
```

因此“健康节点在原切片中的位置”自然适合返回 `Vec<usize>`。这里返回的是原始节点切片中的
索引，不是新的健康节点列表里的相对位置。

## 放回当前场景

```rust
fn healthy_worker_indices(workers: &[Worker]) -> Vec<usize>
```

所有权关系是：

```text
调用者拥有 Vec<Worker>
        │
        ├─ 临时共享借用 ──> workers: &[Worker]
        │                         │
        │                         └─ 只读取并计算索引
        │
        └─ 调用结束后仍然拥有并可继续使用原 Vec

函数自己创建并返回 Vec<usize>
调用者取得这个索引 Vec 的所有权
```

边界场景中，没有健康节点时可以返回：

```rust
Vec::<usize>::new()
```

也可以在返回类型或比较上下文足够明确时写成：

```rust
Vec::new()
```

## 一句话记忆

```text
&[Worker]：节点是你的，我只借一段来读。
Vec::<usize>::new()：创建空 Vec，并明确告诉编译器元素是 usize。
```
