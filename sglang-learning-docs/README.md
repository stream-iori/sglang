# SGLang 源码学习文档

8 周学习计划，从零开始掌握 SGLang 核心运行时并完成第一次代码贡献。

> **文档版本**: 基于 SGLang commit `b8d7351a74` (2026-06-10)
> SGLang 迭代极快，如果某个函数/文件找不到了，参考下方「代码变更跟踪」章节。
>
> **查看建议**: 文档中大量使用 Mermaid 图表，推荐使用支持 Mermaid 渲染的工具阅读（VS Code + Mermaid 插件、GitHub 在线查看、Typora 等）。

## 目录结构

```
sglang-learning-docs/
├── 01-architecture/           # 架构认知 — 系统全貌，读一遍建立心智模型
│   ├── overview.md            # 总览：架构图、概念映射、8 周计划
│   ├── foundations.md         # Week 1 主线：项目结构、多进程架构、请求生命周期
│   └── week1-detailed.md      # Week 1 详细讲义：每日任务、读码路径、验收标准
├── 02-core-systems/           # 核心子系统 — 按模块深入，最常更新
│   ├── scheduler-and-cache.md # Scheduler 调度、RadixCache、内存池
│   ├── week2-detailed.md      # Week 2 详细讲义：Scheduler 和 RadixCache
│   ├── model-execution.md    # ModelRunner、采样、Continuous Batching
│   └── week3-detailed.md      # Week 3 详细讲义：ForwardBatch、ModelRunner、采样
├── 03-advanced/               # 进阶特性 — 需要多卡，按需阅读
│   ├── speculative-and-distributed.md  # 投机解码、PD 分离、分布式概览
│   ├── multi-gpu.md           # TP/DP/DPA/EP/PP 实战
│   └── pd-disaggregation.md  # PD 分离部署、传输后端、EPD、HiCache
├── 04-practice/               # 动手实操 — 有 GPU 后真正用到
│   ├── exercises.md           # Scheduler/RadixCache/ZMQ 可运行 demo
│   ├── exercise-solutions.md  # 练习答案与参考实现方向
│   ├── first-code-change.md   # 最小改代码路径
│   ├── server-and-benchmark.md # Server 启动、Benchmark、性能调优
│   ├── profiling.md           # Profiling、故障排查、PR 分析
│   └── testing-and-ci.md     # 测试体系、CI/CD、第一次代码修改
├── 05-reference/              # 工具箱 — 无序，遇到问题时翻
│   ├── glossary.md            # 高频术语表
│   ├── prerequisites.md       # 前置知识：Python 构建 + Transformers + 数学
│   ├── performance-intuition.md # GPU 带宽/算力、KV Cache 计算、napkin math
│   ├── debugging-guide.md    # pdb、VS Code、日志、py-spy、常见报错
│   └── faq.md                # ScheduleBatch vs ForwardBatch、lock_ref 等
├── setup/                     # 环境搭建 — 一次性，搭完不再看
│   ├── mac-debug.md           # Mac 调试环境
│   ├── gpu-setup.md           # GPU 环境搭建
│   └── setup_mac.sh          # Mac 环境一键搭建脚本
├── 06_demo_scheduler.py       # Scheduler 主循环 demo
├── 06_demo_radix_cache.py     # 前缀缓存 demo
├── 06_demo_zmq_pipeline.py    # ZMQ 流水线 demo
├── capstone/                  # 毕业项目
│   └── capstone.md           # 毕业项目、提交 PR、自我评估
└── conftest.py               # Mac 兼容层 (triton/sgl_kernel/torch.mps stub)
```

## 学习路径

### 路径 A: 完整 8 周（推荐）

Phase 1 — 源码理解 (Mac, Week 1-4, 每天 2h)

| 周 | 文档 | 内容 |
|---|---|---|
| Day 0 | [setup/mac-debug.md](./setup/mac-debug.md) | 搭建 Mac 调试环境 |
| 按需 | [05-reference/prerequisites.md](./05-reference/prerequisites.md) | 前置知识补充 |
| W1 | [01-architecture/overview.md](./01-architecture/overview.md) → [foundations.md](./01-architecture/foundations.md) → [week1-detailed.md](./01-architecture/week1-detailed.md) | 架构全貌、启动链路、请求生命周期 |
| W2 | [02-core-systems/scheduler-and-cache.md](./02-core-systems/scheduler-and-cache.md) → [week2-detailed.md](./02-core-systems/week2-detailed.md) | Scheduler + RadixCache |
| W3 | [02-core-systems/model-execution.md](./02-core-systems/model-execution.md) → [week3-detailed.md](./02-core-systems/week3-detailed.md) | ModelRunner + 采样 |
| W4 | [03-advanced/speculative-and-distributed.md](./03-advanced/speculative-and-distributed.md) | 投机解码、PD 分离、分布式概览 |

Phase 2 — GPU 实战 (NVIDIA GPU, Week 5-8, 每天 2h)

| 周 | 文档 | 内容 |
|---|---|---|
| Day 20 | [setup/gpu-setup.md](./setup/gpu-setup.md) | 搭建 GPU 环境 |
| W5 | [04-practice/server-and-benchmark.md](./04-practice/server-and-benchmark.md) | Server 启动、Benchmark |
| W6 | [04-practice/profiling.md](./04-practice/profiling.md) | Profiling、故障排查 |
| W7 | [04-practice/testing-and-ci.md](./04-practice/testing-and-ci.md) | 测试体系、第一次代码修改 |
| W8 | [capstone/capstone.md](./capstone/capstone.md) | 毕业项目、提交 PR |

### 路径 B: 快速上手（已有 LLM serving 经验）

1. [01-architecture/overview.md](./01-architecture/overview.md) — 10 分钟了解架构
2. [02-core-systems/scheduler-and-cache.md](./02-core-systems/scheduler-and-cache.md) — 核心调度逻辑
3. [05-reference/faq.md](./05-reference/faq.md) — 快速解惑
4. 直接看感兴趣的子系统源码

### 路径 C: 只想贡献代码

1. [setup/mac-debug.md](./setup/mac-debug.md) — 搭环境
2. [01-architecture/foundations.md](./01-architecture/foundations.md) — 了解项目结构
3. [04-practice/testing-and-ci.md](./04-practice/testing-and-ci.md) — 测试和 CI 流程
4. [capstone/capstone.md](./capstone/capstone.md) — PR 流程参考

## 快速开始

### Mac 环境

```bash
# 1. 搭建 Mac 调试环境 (一键)
bash sglang-learning-docs/setup/setup_mac.sh

# 2. 验证: 跑单测
source python/.venv/bin/activate
PYTHONPATH="sglang-learning-docs:python" python -m pytest test/registered/unit/entrypoints/openai/test_protocol.py -v
```

### GPU 环境

进入 Phase 2 前，按 [setup/gpu-setup.md](./setup/gpu-setup.md) 搭建 GPU 环境。

## 配合阅读的辅助材料

以下文档不属于主线，随时查阅：

| 文档 | 用途 | 何时阅读 |
|---|---|---|
| [01-architecture/week1-detailed.md](./01-architecture/week1-detailed.md) | Week 1 每日读码路径、命令、验收标准 | Week 1 主线 |
| [02-core-systems/week2-detailed.md](./02-core-systems/week2-detailed.md) | Week 2 每日读码路径、Scheduler/RadixCache 图解 | Week 2 主线 |
| [02-core-systems/week3-detailed.md](./02-core-systems/week3-detailed.md) | Week 3 每日读码路径、ForwardBatch/采样图解 | Week 3 主线 |
| [04-practice/exercises.md](./04-practice/exercises.md) | Scheduler/RadixCache/ZMQ 可运行 demo | Week 1-2 配合主线 |
| [04-practice/exercise-solutions.md](./04-practice/exercise-solutions.md) | 练习答案、参考实现方向、验收标准 | 做完练习后 |
| [04-practice/first-code-change.md](./04-practice/first-code-change.md) | 从改 demo 到补测试的最小路径 | Week 3 后 |
| [05-reference/glossary.md](./05-reference/glossary.md) | SGLang 高频术语速查 | 随时 |
| [05-reference/performance-intuition.md](./05-reference/performance-intuition.md) | GPU 带宽/算力、napkin math | Week 2 后 |
| [05-reference/debugging-guide.md](./05-reference/debugging-guide.md) | pdb、日志、py-spy | 遇到问题时 |
| [03-advanced/multi-gpu.md](./03-advanced/multi-gpu.md) | TP/DP/EP/PP 多卡实战 | 毕业后 + 多卡 |
| [03-advanced/pd-disaggregation.md](./03-advanced/pd-disaggregation.md) | PD 分离部署深入 | 毕业后 + 多卡 |

## 已验证可运行的单测

| 测试 | 路径 | 关联知识点 |
|---|---|---|
| OpenAI API 协议 | `test/registered/unit/entrypoints/openai/test_protocol.py` | 请求/响应数据结构 |
| RadixCache 前缀缓存 | `test/registered/unit/mem_cache/test_radix_cache_unit.py` | RadixKey, TreeNode, 缓存淘汰 |
| 交互式探索数据结构 | (Python REPL) | SamplingParams, ForwardMode, Req |

## 代码变更跟踪

SGLang 每周合入数十个 PR。如果发现文档中提到的函数名/行号与实际代码不符：

```bash
# 查看某个关键文件的变更历史
git log --oneline -20 -- python/sglang/srt/managers/scheduler.py

# 找到函数被重命名/移动的位置
git log --oneline --all -S "event_loop_normal" -- "*.py"

# 对比文档基准版本与当前版本的差异
git diff b8d7351a74..HEAD -- python/sglang/srt/managers/scheduler.py
```

### 相对稳定的核心接口 (不太会变)

- `Scheduler.event_loop_normal()` / `event_loop_overlap()` — 主循环结构
- `RadixCache.match_prefix()` / `insert()` / `evict()` — 缓存三板斧
- `ModelRunner.forward()` — 前向入口
- `ForwardMode` enum — 前向模式枚举
- ZMQ 多进程通信架构 (进程角色和通信拓扑)

### 经常重构的部分 (留意变化)

- `scheduler_components/` — 持续拆分和新增组件
- `model_executor/` — FlashInfer/Attention backend 快速演进
- `speculative/` — 投机解码策略持续优化
- `disaggregation/` — PD 分离架构快速迭代
- Server args / CLI 参数 — 经常新增和重命名
