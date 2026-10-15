# 协变、逆变与不变：从 Java 泛型理解 Rust 生命周期

[返回索引](index.md) · 前置：[所有权、转移与借用](ownership-move-and-borrowing.md)

## 先抓住一个问题

**里面的类型可以替换，外面的容器能不能跟着替换？这就是变型（variance）研究的问题。**

本文用 `A → B` 表示：需要 `B` 的地方，可以提供 `A`。箭头表示子类型替换，不代表执行任意类型转换。

| 关系 | 已知 `A → B`，套上 `F` 后怎样 | 大白话 |
|---|---|---|
| 协变 | `F<A> → F<B>` | 同方向替换 |
| 逆变 | `F<B> → F<A>` | 反方向替换 |
| 不变 | 不能由此推出上述任一方向 | 不能跟着替换参数 |

推荐按顺序读：

```text
Java：猫和动物 → extends / super → Rust：长引用和短引用
                                            │
                                            v
                             Vec<T> → &T / &mut T → 两层引用
```

下面的代码都是独立语法示例，不是 my-smg 的实现任务。标注“拒绝”的片段用于展示类型检查边界。

## Java：先区分猫和猫列表

假设有以下类型；示例中的列表使用 `java.util.List` 和 `java.util.ArrayList`：

```java
class Animal {}
class Cat extends Animal {}
class Dog extends Animal {}
```

```text
Animal
├── Cat
└── Dog

Cat → Animal：需要动物时，可以提供猫。
```

### 不变：List<Cat> 不是 List<Animal>

```java
List<Cat> cats = new ArrayList<>();
List<Animal> animals = cats; // 拒绝
```

假如允许，两个变量就会指向同一个列表：

```text
cats ──────┐
           ├──> 原本只能保存猫的列表
animals ───┘
```

接下来 `animals.add(new Dog())` 就能往猫列表里放狗，而 `cats.get(0)` 仍按猫使用。
因此 Java 的 `List<T>` 对 `T` 不变。它本身的继承关系不受影响，例如 `ArrayList<Cat>` 仍可赋给 `List<Cat>`。

### 协变视角：extends，把列表作为动物来源

```java
List<Cat> cats = new ArrayList<>();
cats.add(new Cat());

List<? extends Animal> source = cats; // 可以
Animal animal = source.get(0);       // 可以

source.add(new Cat()); // 拒绝
source.add(new Dog()); // 拒绝
```

`? extends Animal` 表示：元素类型是某个确定但当前未知的 Animal 子类型，也可能就是 Animal。

| 操作 | 是否允许 | 原因 |
|---|---|---|
| 读取为 `Animal` | 可以 | 无论里面是哪种子类型，都属于动物 |
| 添加一只猫 | 不可以 | 背后可能是 `List<Dog>` |
| 添加一只狗 | 不可以 | 背后可能是 `List<Cat>` |

这里限制的是添加具体动物；`null` 是特殊情况。列表也不是完全只读，例如仍可能支持 `clear()`。

### 逆变视角：super，把列表作为猫的接收处

```java
List<Animal> animals = new ArrayList<>();
animals.add(new Dog());

List<? super Cat> destination = animals; // 可以
destination.add(new Cat());             // 可以

Object value = destination.get(0); // 可以
Cat cat = destination.get(0);      // 拒绝
```

`List<Cat>`、`List<Animal>`、`List<Object>` 都可以赋给 `List<? super Cat>`，它们都能接收猫。
但是里面原有的值未必是猫，所以读取时只能保证是 Object。

方向可用 Cat 和 Animal 两种边界对照：

```text
Cat → Animal

协变：List<? extends Cat> → List<? extends Animal>
逆变：List<? super Animal> → List<? super Cat>
```

### PECS：来源 extends，去向 super

```java
static <T> void copy(
    List<? extends T> source,
    List<? super T> destination
) {
    for (T value : source) {
        destination.add(value);
    }
}
```

```text
source                  destination
读取 T ─────── T ───────> 写入 T
extends                  super
```

Java 的 `List<T>` 本身仍然不变；通配符是在使用位置表达边界。不要把 `? extends` 当成列表实现改变了。

## Rust：生命周期提供了另一种替换关系

Java 常用子类和父类演示子类型；Rust 没有类继承，最常见的例子是引用的有效范围。

假设 `'long` 至少和 `'short` 一样长，Rust 写作 `'long: 'short`：

```text
&'long str → &'short str

一个长期有效的引用，可以交给只需要短期使用的地方。
一个短期有效的引用，不能承诺长期有效。
```

生命周期是编译期的有效范围约束，不是秒数，不会延长数据的存活时间。

```rust
fn print(text: &str) {
    println!("{text}");
}

fn main() {
    let text: &'static str = "hello";
    print(text); // 长期有效的引用，可以只在本次调用中使用
}
```

### &'static T 和 T: 'static 有什么区别

| 写法 | 约束什么 | 不代表什么 |
|---|---|---|
| `&'static str` | 引用指向的字符串在整个程序期间有效 | 保存引用的局部变量必须一直存在 |
| `T: 'static` | 类型 T 不依赖短生命周期的借用 | T 的值永远不能被销毁 |

拥有数据的 `String` 通常满足 `T: 'static`，但仍会在离开作用域时正常释放。
局部 String 的 `&String` 则不能仅靠加类型标注变成 `'static` 引用。

## Rust 泛型也有变型

**“对泛型参数 T 协变”和“通过生命周期举例”可以同时成立：T 本身可以是引用类型。**

例如：

```rust
fn shorten_vec<'a>(values: Vec<&'static str>) -> Vec<&'a str> {
    values
}
```

```text
参数 T：     &'static str →     &'a str
                    │              │
                    v              v
整个类型：Vec<&'static str> → Vec<&'a str>
```

没有逐个重新创建字符串；这里发生的是满足约束的类型替换。`Vec<T>` 对 T 协变。

### 常见类型对 T 的关系

| 类型 | 对 T 的关系 | 关键点 |
|---|---|---|
| `Option<T>` | 协变 | 元素的子类型关系可以传递到外层 |
| `Vec<T>` | 协变 | 同上 |
| `Box<T>` | 协变 | 同上 |
| `&T` | 协变 | 不能通过这个共享引用直接替换 T |
| `&mut T` | 不变 | 可以往原位置写入新的 T |
| `Cell<T>` | 不变 | 可以通过共享引用替换其中的值 |
| `fn(T)` | 逆变 | T 出现在函数输入位置 |
| `fn() -> T` | 协变 | T 出现在函数输出位置 |

不能简单记成“可修改的容器都不变”：`Vec<T>` 自身协变，但通过 `&mut Vec<T>` 修改时，
外层 `&mut` 会要求里面的整个 `Vec<T>` 保持不变。

用户定义结构体的变型由字段使用参数的位置推导；同一参数同时出现在协变和逆变位置时，会成为不变。

## 为什么 &mut T 对 T 不变，却对外层生命周期协变

先固定 T 为 i32，只比较外层借用时间：

```text
&'long i32     → &'short i32       可以
&'long mut i32 → &'short mut i32   可以
```

下面两个函数都能编译：

```rust
fn shorten<'long: 'short, 'short>(x: &'long i32) -> &'short i32 {
    x
}

fn shorten_mut<'long: 'short, 'short>(x: &'long mut i32) -> &'short mut i32 {
    x
}
```

两边指向的数据类型始终是 i32，变化的是“交出去借用多久”。日常代码中的可变引用短期使用也常通过重新借用完成。

要演示里面的 T 能否变化，就需要让 T 自身包含生命周期。两层引用因此出现。

### 先读懂 &&str 和 *slot

```rust
let text: &str = "hello";
let slot: &&str = &text;
```

```text
slot ──引用──> text ──引用──> "hello"
 &&str          &str            str
                 ↑
               *slot
```

| 表达式 | 类型 | 含义 |
|---|---|---|
| `text` | `&str` | 字符串引用 |
| `&text` / `slot` | `&&str` | 对这个字符串引用再借用一次 |
| `*slot` | `&str` | 解开外层，取得里面的引用值 |
| `**slot` | `str` | 再解一层，得到字符串切片 place；不能直接作为普通局部值搬走 |

`*slot` 取得的是可复制的共享引用，不会复制或释放字符串内容。
`println!("{}", slot)` 和 `println!("{}", *slot)` 在这里都能显示 hello，格式化支持引用层层转发。

### 把两个生命周期分别标出来

```text
&'outer mut &'inner str
└─────────┘ └────────┘
 外层借用       T

'outer：借用“保存字符串引用的变量”多久
'inner：里面的字符串引用有效多久
```

`&&'a str` 可以展开理解为 `&'outer &'a str`：外层也有生命周期，只是在原写法里省略了。
合法的两层借用还要求里面的数据在外层使用期间有效。

### 共享引用：里面的长引用可以按短引用读取

```rust
fn read<'a>(slot: &&'a str, local: &'a str) {
    println!("{} / {}", *slot, local);
}

fn main() {
    let text: &'static str = "hello";
    let local = String::from("world");
    read(&text, &local); // 可以
}
```

`read` 需要里面的字符串在 `'a` 期间有效；hello 是静态字符串，当然满足。
两个参数里的 `'a` 表示共同满足的使用范围，不要求两个原始值恰好同时创建和销毁。

```text
& &'static str → & &'a str
  └─────────┘     └─────┘
       T 可以跟着缩短
```

这里只能读取，无法通过 slot 把一个短引用写回 text。因此 `&T` 对 T 协变。

普通打印函数用 `&str` 即可；这里用 `&&str`，只是为了展示外层引用对内层类型的关系。

### 可变引用：不能把长期引用槽位当成短期引用槽位

```rust
fn replace<'a>(slot: &mut &'a str, value: &'a str) {
    *slot = value;
}

fn main() {
    let mut text: &'static str = "hello";
    {
        let local = String::from("world");
        replace(&mut text, &local); // 拒绝：local 不满足 'static
    }
    println!("{text}");
}
```

假如允许：

```text
text 声明只能保存 'static 引用
          │
          │ 假如 &mut &'static str 能当成 &mut &'a str
          v
replace 把 &local 写进 text
          │
          v
local 被销毁，text 却还存在
          │
          v
再次读取 text，引用已经悬空
```

因此 `&mut T` 对 T 不变。反方向也不成立：短期有效的引用不能当作长期引用读取。

| 类型 | 改变的位置 | 结果 |
|---|---|---|
| `&'a T` | 外层 `'a` | 协变，可以缩短外层借用时间 |
| `&'a T` | 内层 T | 协变，允许 T 的子类型替换 |
| `&'a mut T` | 外层 `'a` | 协变，可以缩短外层借用时间 |
| `&'a mut T` | 内层 T | 不变，防止写回不符合原约束的值 |

## Rust 的逆变：看函数参数

仍假设 `'long: 'short`，比较两个已经确定生命周期的函数指针类型：

```text
&'long str → &'short str

fn(&'short str) → fn(&'long str)
```

一个能处理短期引用的函数，也能处理长期引用；一个只能处理长期引用的函数，不能随便接收短期引用。

```rust
fn adapt<'long: 'short, 'short>(
    handler: fn(&'short str),
) -> fn(&'long str) {
    handler
}
```

输入越宽松，函数越能替代输入要求更严格的函数，所以输入位置是逆变。
返回值则相反：需要短期引用时，返回长期引用的函数可以胜任，所以输出位置协变。

这里有意写出生命周期。裸写 `fn(&str)` 涉及更高阶生命周期，不能机械当作某个固定的 `fn(&'short str)`。

## Rust trait 约束不是 Java 的类继承

| Java | Rust | 能否直接照搬 |
|---|---|---|
| `Cat extends Animal` | 无对应的类继承 | 不能 |
| `T extends Animal` 泛型约束 | `T: Animal` trait 约束 | 能类比“要求能力”，不能类比所有子类型规则 |
| `? extends T` / `? super T` | 无直接对应通配符 | 不能逐字翻译 |
| 使用通配符表达边界 | 按字段推导类型的变型 | 机制不同 |

例如以下 trait 只说明 Dog 实现了某种能力：

```rust
trait Animal {}
struct Dog;
impl Animal for Dog {}
```

`Box<Dog>` 可以强制转换成 `Box<dyn Animal>`，但这不是 Java 式子类关系，也不能自动套用到 Vec：

```rust
let dogs: Vec<Box<Dog>> = vec![Box::new(Dog)];
let animals: Vec<Box<dyn Animal>> = dogs; // 拒绝
```

需要逐个转换：

```rust
let animals: Vec<Box<dyn Animal>> = dogs
    .into_iter()
    .map(|dog| dog as Box<dyn Animal>)
    .collect();
```

这不否定 `Vec<T>` 协变：协变传播的是已有的子类型关系，而不是任意强制转换关系。
Rust 的子类型关系还包含更高阶生命周期相关情况，不能说“只有生命周期长短”这一种。

## 回到讨论起点：issue #164250

这个问题展示的是“类型约束检查遗漏”，不是协变本身不安全。

```text
结构体要求 T: 'static
         │
先构造空容器，随后字段赋值
         │
编译器漏掉相关生命周期约束检查
         │
局部引用被放进容器
         │
Drop 仍相信 T: 'static，把引用保存到更长寿的位置
         │
原数据销毁后再次使用 → 悬空引用
```

此前对原示例做过只编译验证：

| 写法 | Rust 1.85.0 / 1.90.0 / 1.96.0 |
|---|---|
| `Q(None)` 后通过字段赋值放入局部引用 | 错误放行 |
| 直接 `Q(Some(&local))` 构造 | 正确拒绝 |
| 显式声明 `Q<&'static String>` 后放入局部引用 | 正确拒绝 |

这是历史版本的验证记录，不代表当前所有工具链的状态。理解时重点记住：
**生命周期可以按规则缩短，但得到的整个类型仍必须满足自己声明的约束。**

讨论与修复入口：[issue #164250](https://github.com/rust-lang/rust/issues/164250)、
[PR #164277](https://github.com/rust-lang/rust/pull/164277)。

## 常见误解

| 误解 | 更准确的理解 |
|---|---|
| 加 `'static` 能让局部值一直存在 | 生命周期标注提出约束，不延长数据存活 |
| Rust 只有生命周期有变型，泛型没有 | 泛型对 T 也有变型，T 可以包含生命周期 |
| `Dog: Animal` 所以 `Vec<Dog>` 能转成动物列表 | trait 实现不建立这种子类型关系 |
| `&mut T` 不变，所以可变借用不能缩短 | 不变的是 T，外层借用生命周期协变 |
| `&&str` 是理解协变的起点 | 它只适合演示内外两层参数，先理解一层引用更容易 |
| `extends` 意味着列表完全不可修改 | 它限制能安全写入的元素类型，不保证只读 |

## 参考与延伸

- [Rust Reference：Subtyping and variance](https://doc.rust-lang.org/reference/subtyping.html)
- [Rust Nomicon：Subtyping and Variance](https://doc.rust-lang.org/nomicon/subtyping.html)
- [Java：Generics, Inheritance, and Subtypes](https://docs.oracle.com/javase/tutorial/java/generics/inheritance.html)
- [Java：Upper Bounded Wildcards](https://docs.oracle.com/javase/tutorial/java/generics/upperBounded.html)
- [Java：Lower Bounded Wildcards](https://docs.oracle.com/javase/tutorial/java/generics/lowerBounded.html)
- 本目录：[Formatter 的两层借用与生命周期](formatter-lifetimes-display-and-error.md)
- 本目录：[Trait、静态分发、动态分发与智能指针](traits-dispatch-and-smart-pointers.md)
