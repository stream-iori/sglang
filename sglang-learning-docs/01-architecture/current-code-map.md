# 10 月源码大改造：学习入口怎么变了

> 源码基准：learning `2aa70e3eb4`，已合入 origin/main `6fc8d9da32`（2026-10-10）。

这次重写以当前检出为准，不再以 6 月讲义的行号和类关系推断执行路径。

| 改造 | 现在的入口 | 对学习的影响 |
|---|---|---|
| 配置命名空间与进程状态 | runtime_context.py、arg_groups/ | 分清原始参数、解析后配置、运行时 flags/resources |
| 统一前缀树 | registry.py → UnifiedRadixCache → TreeCore + components | Full/SWA/Mamba 在同一树上协作；不再只读旧 RadixCache |
| KV 分配与地址表达 | allocation.py、allocator/、KVLocPlan | token ID、请求位置、虚拟槽位、物理槽位要分开 |
| 执行器拆分 | runner/ 的 eager、decode graph、prefill graph | forward 是路由入口；prefill 也可能用 Graph |
| ModelRunner 初始化拆分 | model_runner_components/ | 权重、Attention、Graph、KV 初始化分别追踪 |
| Scheduler 组件化 | scheduler_components/ | 输出、指标、内存观察等不只在 scheduler.py 里 |
| Rust 服务链 | sglang-server / processor / renderer / radix-tree | Rust 是多个职责不同的模块，不代表 Python Scheduler 被替换 |
| 统一 kernel 门面 | python/sglang/kernels/ | 旧 jit_kernel 路径已移除；按算子组定位 |
| 测试重新组织 | test/registered/ + register_*_ci | 文件目录表达主题，注册信息表达 CI 硬件和阶段 |
| 标准 MPS | hardware_backend/mps/runtime.py | 本地学习可走标准 Torch ModelRunner |

## 两处容易读错

```text
“有旧文件” ≠ “当前默认实例就是旧类”
    先查 registry 的构造分支，再看日志/对象类型。

“README 描述目标目录” ≠ “当前文件全部已迁移”
    allocator/ 已存在；pool/、pool_host/ 尚未建立。
    物理池仍从 memory_pool.py 等实际文件进入。
```

统一树支持 Python/Rust backend，具体选哪个由 tree_core_registry 决定；存在 Rust crate 不能证明本地已启用它。

## 更新源码以后怎样复核

```bash
git rev-parse HEAD
git log -1 origin/main
git log --oneline -10 -- python/sglang/srt/mem_cache
rg -n 'default_radix_cache_factory|create_unified_radix_cache' python/sglang/srt/mem_cache/registry.py
rg -n 'can_run_graph|eager_runner.execute' python/sglang/srt/model_executor/model_runner.py
```

“最新”在此表示已经合并到本地的 `6fc8d9da32`，本次文档工作没有再次 fetch 或更新代码。

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/runtime_context.py](../../python/sglang/srt/runtime_context.py) | 配置分层 |
| [python/sglang/srt/mem_cache/registry.py](../../python/sglang/srt/mem_cache/registry.py) | 默认缓存选择 |
| [python/sglang/srt/mem_cache/README.md](../../python/sglang/srt/mem_cache/README.md) | 分层设计与实际迁移状态 |
| [python/sglang/srt/mem_cache/unified_cache/tree_core_registry.py](../../python/sglang/srt/mem_cache/unified_cache/tree_core_registry.py) | 树 backend 选择 |
| [python/sglang/kernels/README.md](../../python/sglang/kernels/README.md) | 新的算子组织 |
| [test/README.md](../../test/README.md) | 当前测试规范 |
