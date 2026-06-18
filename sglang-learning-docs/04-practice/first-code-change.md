# 最小改代码路径：从能读到能改

> 目标：给只会基础 Python 的学习者一条低风险路径。先改 demo，再改测试，再碰 SGLang 核心代码。

## 总路线

```mermaid
flowchart LR
    A["读 Week1-3"] --> B["改 demo"]
    B --> C["跑 demo"]
    C --> D["读对应源码"]
    D --> E["改一个单测"]
    E --> F["跑单测"]
    F --> G["准备真实 PR"]

    style B fill:#74b9ff,color:#000
    style E fill:#ffa502,color:#000
    style G fill:#7bed9f,color:#000
```

## 1. 不要一上来改核心 Scheduler

| 阶段 | 推荐改什么 | 不推荐改什么 |
|---|---|---|
| 第 1 次 | demo 脚本、文档 typo | `scheduler.py` 主逻辑 |
| 第 2 次 | 单测、协议字段测试 | CUDA kernel |
| 第 3 次 | 小工具函数、边界条件 | 多卡/PD/投机解码 |

原因很简单：

```text
核心代码影响面大。
初学者先练“改动 -> 验证 -> 回滚”的闭环。
```

## 2. 三个入门任务

| 任务 | 文件 | 难度 | 验证 |
|---|---|---:|---|
| 给 Scheduler demo 加中途请求 | `06_demo_scheduler.py` | L1 | 运行 demo |
| 给 Radix demo 加 hit ratio | `06_demo_radix_cache.py` | L1 | 运行 demo |
| 给 OpenAI protocol 加测试 | `test/registered/unit/entrypoints/openai/test_protocol.py` | L2 | pytest |

## 3. 任务 A：Scheduler demo 加中途请求

目标：模拟真实 continuous batching。

改动点：

```text
step == 2 时，插入 req-3。
观察 req-3 先 EXTEND，再进入 DECODE。
```

验收输出应包含：

```text
[recv] rid=req-3
[run] mode=extend
[run] mode=decode
```

验证：

```bash
python sglang-learning-docs/06_demo_scheduler.py
```

## 4. 任务 B：Radix demo 加 hit ratio

目标：每次 prefix match 后打印命中率。

公式：

```text
hit_ratio = len(matched) / len(input_tokens)
```

示例输出：

```text
input=[1,2,3,4,5,99] matched=[1,2,3,4,5] hit_ratio=0.83
```

验证：

```bash
python sglang-learning-docs/06_demo_radix_cache.py
```

## 5. 任务 C：补一个协议单测

先读：

```bash
sed -n '1,220p' test/registered/unit/entrypoints/openai/test_protocol.py
```

建议只加这类低风险测试：

| 测试类型 | 例子 |
|---|---|
| 字段默认值 | 某字段缺省时是否为预期默认 |
| 字段校验 | 非法值是否抛错 |
| 序列化 | 请求对象能否正确 dump |

验证：

```bash
PYTHONPATH="python" python/.venv/bin/python -m pytest \
  test/registered/unit/entrypoints/openai/test_protocol.py -v
```

## 6. 读源码模板

每读一个函数，只记录 5 件事：

| 项 | 问题 |
|---|---|
| 输入 | 参数是什么？来自哪里？ |
| 输出 | return 什么？发给谁？ |
| 状态变化 | 修改了哪些 `self.xxx`？ |
| 下游调用 | 调了哪些关键函数？ |
| 失败路径 | 什么时候 abort/error/return None？ |

模板：

```markdown
## 函数：xxx

| 项 | 记录 |
|---|---|
| 输入 | |
| 输出 | |
| 状态变化 | |
| 下游调用 | |
| 失败路径 | |
```

## 7. 提交前检查

```bash
# 看自己改了什么
git diff -- sglang-learning-docs test/registered/unit/entrypoints/openai/test_protocol.py

# 跑相关验证
python -m py_compile sglang-learning-docs/06_demo_scheduler.py sglang-learning-docs/06_demo_radix_cache.py
PYTHONPATH="python" python/.venv/bin/python -m pytest \
  test/registered/unit/entrypoints/openai/test_protocol.py -v
```

## 8. 合格标准

| 能力 | 合格表现 |
|---|---|
| 改 demo | 改完能运行，输出能解释 |
| 改测试 | 能只跑相关测试，不跑全量 |
| 看 diff | 能说清每一行改动目的 |
| 写 commit | commit message 短且具体 |
| 控制范围 | 不夹带无关格式化和重构 |

