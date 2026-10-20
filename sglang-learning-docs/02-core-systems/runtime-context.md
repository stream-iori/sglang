# RuntimeContext：参数、状态与资源怎么分开

> 源码基准：上游 `16eb87c06f`；本地集成 `4f0e1cbea2`（2026-10-20）。本地日志增强与验证范围见总索引和 Mac 验证记录。

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

## 新增：模型拥有 context，执行时串行切换

当前 `create_context(server_args, role=..., ranks=...)` 构造一份独立 RuntimeContext，但不替换进程当前配置。`use_context(context)` 在作用域内切换 `_CONTEXT` 和 `_PARALLEL`，退出时按对象身份恢复。

```text
当前进程 context A
    ├─ create_context(...) → B，A 仍激活
    ├─ with use_context(B): → B 激活，串行执行模型工作
    └─ 退出作用域          → A 恢复
```

| 机制 | 隔离范围 | 重要限制 |
|---|---|---|
| publish | 配置投影到当前 context | 再发布会重建 bags，并影响 override 记录 |
| create_context | 新建模型拥有的配置/状态容器 | 不自动激活，也不自动隔离所有外部进程资源 |
| use_context | 临时切换进程当前 context | 修改进程全局引用，调用者必须串行化执行 |
| get_forward().scoped | 本轮动态字段，contextvar 作用域 | 不能代替整个模型上下文切换 |

不能因为用了 contextmanager 就认为它是 Java ThreadLocal：实现明确不提供并发线程隔离。嵌入式模型上下文的存在，也不意味着多个模型可随意同时使用共享 process groups 或其他外部资源。

## 配置变化怎样取证

| 想证明的事 | 应读位置 |
|---|---|
| 用户传过什么 | 原始 ServerArgs / 启动记录 |
| 设备规则最终决定什么 | resolving_view / 发布后的 namespace bag |
| 启动后哪些字段被覆盖 | get_context().overrides_log()、config_leaf |
| 某个模型私有选择是什么 | model/runner 构造参数或属性；例如 shared_experts_config |
| 本次 forward 临时设置什么 | ForwardContext、get_forward 作用域和恢复边界 |

当前 shared-experts 相关配置进一步归属模型，`ModelRunner._forward_raw` 在模型配置作用域中执行。不能把旧的全局融合 flag 当作所有模型通用的唯一决策点。

## 读码练习

```bash
rg -n 'def publish|def create_context|def use_context|def _configure_context' python/sglang/srt/runtime_context.py
rg -n 'create_context|use_context' test/registered/unit python/sglang/srt
rg -n 'shared_experts_config|def _forward_raw' python/sglang/srt/model_executor/model_runner.py
```

验收：解释为什么修改原始 ServerArgs、覆盖 bag、切换 context 是三种动作。答案应分别包含启动来源、有效配置修改与进程当前配置引用变化，不能只说“都是改全局变量”。

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/runtime_context.py](../../python/sglang/srt/runtime_context.py) | publish、配置 bags、flags/resources/forward |
| [python/sglang/srt/arg_groups](../../python/sglang/srt/arg_groups) | 参数组织 |
| [test/registered/unit/test_runtime_context.py](../../test/registered/unit/test_runtime_context.py) | 进程状态与隔离 |
| [test/registered/unit/test_runtime_context_override.py](../../test/registered/unit/test_runtime_context_override.py) | 覆盖语义 |
