# 算子入口：sglang.kernels、JIT 与 AOT

> 源码基准：learning `2aa70e3eb4`，已合入 origin/main `6fc8d9da32`（2026-10-10）。

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

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/kernels/README.md](../../python/sglang/kernels/README.md) | 当前组织与选择语义 |
| [python/sglang/kernels/registry.py](../../python/sglang/kernels/registry.py) | 元数据注册 |
| [python/sglang/kernels/selector.py](../../python/sglang/kernels/selector.py) | 实现选择 |
| [python/sglang/kernels/jit](../../python/sglang/kernels/jit) | JIT 基础设施 |
| [test/registered/kernels](../../test/registered/kernels) | 测试/benchmark |
