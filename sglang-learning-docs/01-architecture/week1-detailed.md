# 第一阶段：启动、请求与进程

> 源码基准：上游 `16eb87c06f`；本地集成 `4f0e1cbea2`（2026-10-20）。本地日志增强与验证范围见总索引和 Mac 验证记录。

按当前源码走，每个阶段以可解释的证据验收，不要求每天固定耗时。

| 步骤 | 动作 | 验收 |
|---|---|---|
| 1 | 运行 Mac 安装和验证 | 保存 server_info 与生成响应 |
| 2 | launch_server → Engine | 画出进程装配图 |
| 3 | OpenAI serving → TokenizerManager | 列出 messages、文本、ID 三个形态 |
| 4 | event_loop_normal | 指出 plan.batch_to_run 和 running_batch |
| 5 | 正常生成与 SSE | 区分 token 到达与文字输出 |

## 配套阅读

- [启动链](foundations.md)
- [分词与增量解码](../05-reference/tokenizer-internals.md)

## 共用观察命令

```bash
rg -n "def event_loop_normal|def get_next_batch_to_run" python/sglang/srt/managers/scheduler.py
rg -n "can_run_graph|eager_runner.execute" python/sglang/srt/model_executor/model_runner.py
```

命令从仓库根目录执行。源码行号由当前搜索获得，避免复制旧版本固定行号。

## 五次练习的明确交付

| 次序 | 操作与阅读 | 留下的材料 | 自测答案要点 |
|---|---|---|---|
| 1 | 安装 → 启动 → verify | 环境、server-info、summary | 安装成功不等于模型可请求 |
| 2 | launch_server → Engine | 进程图，标 IPC 方向 | 子进程不是共享同一个 Python 堆 |
| 3 | 同一问题分别 generate/chat | 原始请求、输入形态差异 | chat template 会增加控制信息 |
| 4 | 给 rid 找入队和输出路径 | 调用链与状态表 | Req 跨轮，batch 每轮变化 |
| 5 | SSE 与最终文本对照 | chunks、拼接结果、DONE | 一个 token 不必对应一个可见字符 |

每次先预测，再运行；预测与实际不同就追源码分支。阶段结束需能从零画出请求到响应的路径，标出每一步处理的是文字、ID、张量还是消息。

卡在 Python 语法时从 experiences 索引按问题补，不需要先学完全部语言特性。

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/managers/scheduler.py](../../python/sglang/srt/managers/scheduler.py) | 主循环 |
| [python/sglang/srt/model_executor/model_runner.py](../../python/sglang/srt/model_executor/model_runner.py) | 执行分支 |
