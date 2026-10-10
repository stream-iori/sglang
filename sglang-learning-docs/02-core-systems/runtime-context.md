# RuntimeContext：参数、状态与资源怎么分开

> 源码基准：learning `2aa70e3eb4`，已合入 origin/main `6fc8d9da32`（2026-10-10）。

当前代码不再只有一个可随意改写的全局 server_args。原始输入、解析后配置、可变 flags、资源句柄和本轮状态分开管理。

```text
CLI / ServerArgs 原始输入
    ↓ resolve / publish
进程内 RuntimeContext
    ├─ 配置 bags：get_exec / get_memory / get_schedule / get_parallel ...
    ├─ get_flags：可变运行状态
    ├─ get_resources：进程资源、句柄
    └─ get_forward：本轮 forward 状态
```

| 入口 | 用来读什么 | 示例问题 |
|---|---|---|
| get_server_args | 原始输入，保留诊断信息 | 用户传入了什么 |
| get_exec / get_memory 等 | 解析后有效配置 | 实际允许哪种 Graph/内存方案 |
| get_parallel | rank、group、并行配置 | 当前进程属于哪个组 |
| get_flags | 可变控制状态 | 当前运行标志是什么 |
| get_resources / get_stream / get_buffer | stream、buffer 等资源 | 谁持有和复用这个句柄 |
| get_forward | 每轮状态 | 这次计算的动态条件 |

## 为什么直接改原始参数可能不生效

```text
原始输入 → 解析并发布 → runner 读取解析后的 bag
    ↑ 此后仅改这里，不能假定已发布的 bag 自动更新
```

读到 `get_exec()` 时，应回到参数声明、解析和 override 路径检查，而不是只打印 `server_args` 就断言生效。

`RuntimeContext.override(source, **fields)` 记录有效配置覆盖；还存在用于限定作用域/诊断等场景的其他 override 接口。它们的范围和恢复语义不同，使用前读对应实现和测试。

## 子进程与测试

每个进程各有自己的 context；父进程中的 Python 全局对象不会变成跨进程共享配置表。启动时发布角色/配置，运行时资源也在相应进程内管理。

单测需要隔离 context，避免上个 case 的 flags、配置、资源污染下一个 case。`snapshot_context`、`restore_context`、`reset_context` 的存在就是读码线索。

读码练习：在 ModelRunner 找一个 `get_exec()` 使用点，追到 `arg_groups/fields/` 的声明，再读 override 单测，说明“输入值”和“有效值”可能在哪一步不同。

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/runtime_context.py](../../python/sglang/srt/runtime_context.py) | publish、配置 bags、flags/resources/forward |
| [python/sglang/srt/arg_groups](../../python/sglang/srt/arg_groups) | 参数组织 |
| [test/registered/unit/test_runtime_context.py](../../test/registered/unit/test_runtime_context.py) | 进程状态与隔离 |
| [test/registered/unit/test_runtime_context_override.py](../../test/registered/unit/test_runtime_context_override.py) | 覆盖语义 |
