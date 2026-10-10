# 第一次改代码：选能完整解释与验证的小问题

> 源码基准：learning `2aa70e3eb4`，已合入 origin/main `6fc8d9da32`（2026-10-10）。

推荐从当前普通路径的可观察问题入手：错误信息、配置校验、输出边界或明确的诊断信息。先写清触发和预期，不为了练习随意加生产日志。

```text
复现 → 找实际执行分支 → 最小修改 → 相关验证 → 检查 diff → commit
```

| 步骤 | 留下的材料 |
|---|---|
| 复现 | 模型、命令、依赖、失败响应或日志 |
| 定位 | 实际类、参数有效值、触发函数 |
| 修改 | 问题与新行为的对应关系 |
| 验证 | 针对问题的最小可靠检查；必要时加回归测试 |
| 审查 | diff 范围、是否误改其他平台 |

## 当前代码组织的约束

| 修改对象 | 先查 |
|---|---|
| Scheduler / ModelRunner | 组件与初始化分层，大类规范 |
| SGLANG_* 环境变量 | environ.py 的定义与访问规范 |
| kernel | kernels/README 与对应算子组 |
| 测试 | test/README、当前注册方式 |
| cache | registry 真实选择、统一组件契约 |

不要仅因 Mac import 失败就删除所有平台相关类：先区分导入耦合和实际执行。此前 MPS 安装保留兼容包，是有真实缺包堆栈支撑的选择。

## 提交前

```bash
git diff --check
git diff --stat
git status --short
```

需要发 PR 时，描述应包含具体问题、新行为、验证和未覆盖范围；不要把学习过程流水账直接当 PR 描述。

## 对照源码

| 文件 | 读什么 |
|---|---|
| [test/README.md](../../test/README.md) | 测试组织 |
| [python/sglang/kernels/README.md](../../python/sglang/kernels/README.md) | kernel 组织 |
| [python/sglang/srt/mem_cache/unified_cache/components/README.md](../../python/sglang/srt/mem_cache/unified_cache/components/README.md) | cache 契约 |
