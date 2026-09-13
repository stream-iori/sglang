# Rust：Cargo Workspace、Package 与 Crate

## 一句话

```text
workspace：统一管理多个 package
package：一个 Cargo.toml 描述的发布/构建单元
crate：一次编译产生的 Rust 代码单元（library 或 binary）
```

典型关系是：

```text
Workspace
├─ Package A（Cargo.toml）
│  ├─ library crate（src/lib.rs）
│  └─ binary crate（src/bin/tool.rs）
└─ Package B（Cargo.toml）
   └─ binary crate（src/main.rs）
```

Workspace 不是把所有源码合并成一个 crate。每个 member 仍是独立 package，拥有自己的依赖、
feature 和编译目标；Workspace 只是让 Cargo 能统一管理它们。

## 为什么使用 Workspace

多个相关 package 放进同一个 Workspace 后，可以共享：

- 一份 `Cargo.lock`，让整个仓库使用一致的依赖解析结果；
- 一个 `target/` 目录，避免每个 package 重复保存构建产物；
- package 元数据和依赖版本等公共配置；
- 根目录中的构建 profile；
- 一组面向全部或指定 member 的 Cargo 命令。

```text
rust/
├─ Cargo.toml       <- workspace root manifest
├─ Cargo.lock       <- 整个 workspace 共用
├─ target/          <- 整个 workspace 共用
├─ sglang-grpc/
│  └─ Cargo.toml    <- member package
├─ sglang-mm/
│  └─ Cargo.toml    <- member package
└─ sglang-server/
   └─ Cargo.toml    <- member package
```

## 根 `Cargo.toml`

当前 SGLang Rust 目录使用的是 virtual workspace：根清单只有 `[workspace]`，没有
`[package]`。

```toml
[workspace]
resolver = "3"
members = [
    "sglang-grpc",
    "sglang-mm",
    "sglang-server",
]
```

源码见 [`rust/Cargo.toml`](../../../rust/Cargo.toml)。

```text
有 [workspace]，没有 [package] -> virtual workspace
有 [workspace]，也有 [package] -> 根 package 本身也是一个 member
```

如果根目录自身也是 package，通常可以省略把 `"."` 写进 `members`；根 package 自动属于
Workspace。`members` 支持 glob，例如 `"crates/*"`，不应加入的目录可放进 `exclude`。

```toml
[workspace]
members = ["crates/*"]
exclude = ["crates/experimental"]
```

## 统一 package 元数据

Workspace 根可以集中声明公共元数据：

```toml
[workspace.package]
version = "0.1.0"
edition = "2024"
license = "Apache-2.0"
```

member 不会自动继承这些值，必须显式选择继承：

```toml
[package]
name = "sglang-mm"
version.workspace = true
edition.workspace = true
license.workspace = true
```

类型变化可以这样理解：

```text
[workspace.package].edition = "2024"   <- 定义公共值
edition.workspace = true               <- member 显式取用
```

因此在根清单中声明一个字段，不等于强制覆盖所有 member。

## 统一依赖版本

根清单可以作为依赖配置的单一来源：

```toml
[workspace.dependencies]
serde = { version = "1", features = ["derive"] }
tokio = { version = "1", features = ["full"] }
```

需要这些依赖的 member 再显式继承：

```toml
[dependencies]
serde = { workspace = true }
tokio = { workspace = true }
```

这里有两个关键点：

1. `[workspace.dependencies]` 不会自动把依赖添加到每个 member；
2. member 可以在继承时追加 `features`，但不能改成另一个 `version`。

```toml
[dependencies]
serde = { workspace = true, features = ["rc"] }
```

最终启用的 feature 还会受到 Cargo feature 合并规则影响，所以“某个依赖在一个 member 中
关闭默认 feature”不代表同一次 Workspace 构建中它一定不会被其他 member 打开。

## member 之间如何依赖

属于同一 Workspace，不代表 package 之间自动可见。`sglang-server` 要使用 `sglang-mm`，仍然
必须正常声明依赖：

```toml
[dependencies]
sglang_mm = {
    package = "sglang-mm",
    path = "../sglang-mm",
    default-features = false,
}
```

上例中三个名字的含义不同：

| 名字 | 含义 |
|---|---|
| `sglang-mm` | 被依赖 package 的名字，即 `[package].name` |
| `../sglang-mm` | 相对于当前 `Cargo.toml` 的本地路径 |
| `sglang_mm` | 当前 crate 在 Rust 代码中使用的依赖名 |

```rust
use sglang_mm::processor::Processor;
```

即使不显式重命名，Cargo 也会把 package 名中的 `-` 转成 Rust 路径中的 `_`；显式写出别名
通常是为了让意图更清楚。

## `resolver` 是什么

`resolver` 选择 Cargo 的依赖与 feature 解析规则。当前 Workspace 使用：

```toml
[workspace]
resolver = "3"
```

它与 Rust 2024 edition 配套。尤其是 virtual workspace 没有根 package，Cargo 无法从
`[package].edition` 推断 resolver，因此应在 `[workspace]` 中显式声明。

要注意：`resolver = "3"` 不等于“不同 member 永远得到完全隔离的 feature 集合”。同一次
构建选择了哪些 package、target 和 feature，仍会影响最终解析结果。排查时先用精确的 package
命令复现，不要只看某个 `Cargo.toml` 作结论。

## profile 只能在根清单生效

构建 profile 应写在 Workspace 根：

```toml
[profile.release]
lto = true
strip = true
opt-level = 3
codegen-units = 1
```

member 清单里的 `[profile.dev]`、`[profile.release]` 会被忽略。若只想调整某个 package，
仍然在根清单写 package override：

```toml
[profile.release.package.sglang-server]
opt-level = 2
```

## 常用命令

在 Workspace 根目录执行：

```bash
# 检查默认选择的 member
cargo check

# 检查所有 member
cargo check --workspace

# 只检查一个 package；-p 是 --package 的缩写
cargo check -p sglang-server

# 检查全部 member，但排除一个
cargo check --workspace --exclude sglang-grpc

# 对指定 package 运行测试
cargo test -p sglang-mm

# 查看 Cargo 解析出的 Workspace、target、依赖和 feature
cargo metadata --no-deps --format-version 1
```

不在 Workspace 根目录时，可以明确指定清单。下面的路径假设命令从 SGLang 仓库根目录执行：

```bash
cargo check \
  --manifest-path rust/Cargo.toml \
  --workspace
```

`-p` 接受的是 `[package].name`，不是目录名、`[lib].name` 或 Python 模块名。

## 默认构建哪些 member

可以用 `default-members` 控制在根目录运行不带 `-p`、`--workspace` 的命令时默认选择谁：

```toml
[workspace]
members = ["service", "tools/*"]
default-members = ["service"]
```

```text
cargo check              -> default-members
cargo check --workspace  -> members 中的全部 package
cargo check -p tool      -> 指定 package
```

对于没有设置 `default-members` 的 virtual workspace，根目录命令默认选择全部 member。

## 如何确认某个 package 属于哪个 Workspace

不要只根据目录层级猜测，直接让 Cargo 给出答案。下面的路径假设命令从 SGLang 仓库根目录
执行：

```bash
cargo metadata \
  --manifest-path rust/Cargo.toml \
  --no-deps \
  --format-version 1
```

重点看：

- `workspace_root`：Workspace 根目录；
- `workspace_members`：所有 member 的 package ID；
- `packages[].manifest_path`：每个 package 对应的清单；
- `target_directory`：共享构建目录。

当前 SGLang 的结果可以简化为：

```text
workspace_root: rust/
target_directory: rust/target/
members:
  - sglang-grpc
  - sglang-mm
  - sglang-server
```

## 常见误区

| 误区 | 实际情况 |
|---|---|
| Workspace 就是一个大 crate | Workspace 管理多个 package；每个 target 仍单独编译成 crate |
| member 可以直接引用另一个 member | 仍需在 `[dependencies]` 中声明 `path` 或版本依赖 |
| 根依赖会自动进入全部 member | member 必须写 `{ workspace = true }` |
| 每个 member 有自己的 lockfile 和 `target/` | Workspace 默认共用根 `Cargo.lock` 和根 `target/` |
| 在 member 中写 profile 能单独生效 | profile 只读取 Workspace 根清单中的配置 |
| `cargo check` 永远检查整个 Workspace | 取决于执行位置、根清单类型和 `default-members` |
| package、crate、module 是同一层概念 | package 是 Cargo 单元，crate 是编译单元，module 是 crate 内代码组织单元 |

## 与模块系统的边界

Workspace 管的是 package 之间的关系；`mod`、`use`、`crate::` 管的是一个 crate 内部的代码
组织。后者见 [`crate`、`mod`、`pub`、`use` 与模块路径](modules-visibility-and-crate.md)。

```text
跨 package：Cargo.toml 的 [dependencies]
跨 module： Rust 代码的 mod / use / crate:: / super::
```

## 一句话记忆

```text
Workspace 共享管理，不共享源码可见性。
根清单提供公共值，member 必须显式继承。
member 互相调用，仍然必须声明依赖。
```
