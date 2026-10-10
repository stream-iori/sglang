# 第一阶段：启动、请求与进程

> 源码基准：learning `2aa70e3eb4`，已合入 origin/main `6fc8d9da32`（2026-10-10）。

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

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/managers/scheduler.py](../../python/sglang/srt/managers/scheduler.py) | 主循环 |
| [python/sglang/srt/model_executor/model_runner.py](../../python/sglang/srt/model_executor/model_runner.py) | 执行分支 |
