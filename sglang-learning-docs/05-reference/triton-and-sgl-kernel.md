# 算子入口：sglang.kernels、JIT 与 AOT

> 源码基准：上游 `16eb87c06f`；本地集成 `4f0e1cbea2`（2026-10-20）。本地日志增强与验证范围见总索引和 Mac 验证记录。

当前公开算子入口按计算职责组织到 `sglang.kernels.ops.<group>`。旧 `sglang.jit_kernel` 已移除，不能继续照旧路径读。

```text
SRT 层调用
   → sglang.kernels.ops.<group> 的门面
   → registry / selector / callable
       ├─ AOT sgl_kernel：提前编译的扩展
       ├─ JIT：按需编译，基础设施在 kernels/jit
       └─ 其他登记的实现 / native reference
```

| 模块 | 负责什么 |
|---|---|
| spec.py | 算子、backend、格式/能力等元数据 |
| registry.py | 注册实现信息，不等于立即导入/编译 |
| selector.py | 按契约解析实现并缓存 callable |
| fused_op.py | 多实现 fused op 合同 |
| ops/ | activation、gemm、attention、kvcache 等职责分组 |
| jit/ | 共享编译/运行基础设施 |

当前 README 明确：多 backend 的 inventory 不等于自动按优先级择优，多个实现需要明确选择。某些门面默认固定 AOT 实现；必须读该 operator 的 wrapper。

## 定位一个算子

```bash
rg -n 'rmsnorm' python/sglang/kernels/ops/layernorm
rg -n 'get_kernel|select_kernel' python/sglang/kernels
rg --files test/registered/kernels/ops | head
```

kernel 测试位于 `test/registered/kernels/ops/<group>/`，benchmark 在对应 benchmark 目录。运行时集成的测试则跟随运行时子系统，不应全部挪到 kernel 目录。

Triton、CUDA JIT 和 MPS 是不同平台/技术路径。本地普通 MPS 不证明这些 CUDA 实现可在 Mac 执行。发生 import 错误时，先查谁在何时导入，而不是随便补一个 CUDA 包。

## 本次上游目录变化怎样判断归属

| 完整计算职责 | 当前归属原则 |
|---|---|
| 跨 rank collective 或 peer-memory | communication，即使融合 GEMM/norm |
| norm 后量化输出 | layernorm，量化是生产者 epilogue |
| KV 内容存取/拷贝/Host transfer | kvcache |
| Attention 索引、page tables、调度 metadata | attention |
| agent-generated 实现 | KDA/Cake backend，仍注册到逻辑算子组 |

这次更新清理了历史合并留下的旧 kernel 文件；不要据旧文件仍在就继续引用。先找 SRT wrapper 的实际 import 和 group 的注册，再找底层实现。

## 练习

选一个 Qwen3 使用的 norm 或 Attention 调用：写出 SRT 调用者、公开 wrapper、实现选择、设备限制、对应测试五项。答案不能停在一个 `.cu` 文件；有源文件不代表当前 Mac 在执行它。

namespace 测试能证明分组和登记契约，不能证明 GPU 数值或吞吐。本轮只执行可在本机运行的 namespace 检查，不宣称 kernel 性能通过。

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/kernels/README.md](../../python/sglang/kernels/README.md) | 当前组织与选择语义 |
| [python/sglang/kernels/registry.py](../../python/sglang/kernels/registry.py) | 元数据注册 |
| [python/sglang/kernels/selector.py](../../python/sglang/kernels/selector.py) | 实现选择 |
| [python/sglang/kernels/jit](../../python/sglang/kernels/jit) | JIT 基础设施 |
| [test/registered/kernels](../../test/registered/kernels) | 测试/benchmark |
