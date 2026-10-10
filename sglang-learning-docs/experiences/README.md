# 读码补充笔记

> 源码基准：learning `2aa70e3eb4`，已合入 origin/main `6fc8d9da32`（2026-10-10）。

这些笔记已按当前学习主线重写。日期文件名只为兼容原链接，内容不是旧日期的运行报告。

| 主题 | 当前笔记 |
|---|---|
| 两个 __init__：包导入与对象初始化 | [2026-06-18-python-init-explained.md](2026-06-18-python-init-explained.md) |
| 对象引用：共享 Req，不等于复制请求 | [2026-06-18-python-vs-java-object-reference.md](2026-06-18-python-vs-java-object-reference.md) |
| 函数签名：位置、关键字与注解 | [2026-06-18-python-function-signature-vs-java.md](2026-06-18-python-function-signature-vs-java.md) |
| 局部 import：等真的需要时才加载 | [2026-06-18-python-local-import.md](2026-06-18-python-local-import.md) |
| PYTHONPATH：确保运行当前检出的源码 | [2026-06-18-pythonpath-explained.md](2026-06-18-pythonpath-explained.md) |
| 找实际实现：从实例构造追到方法 | [2026-06-18-find-abstractmethod-implementation.md](2026-06-18-find-abstractmethod-implementation.md) |
| 插件与 registry：两种注册别混在一起 | [2026-06-18-sglang-plugins-and-entry-points.md](2026-06-18-sglang-plugins-and-entry-points.md) |
| self 和 cls：实例与类 | [2026-06-21-python-self-cls-vs-java-this-factory.md](2026-06-21-python-self-cls-vs-java-this-factory.md) |
| if batch：对象存在与批次非空不同 | [2026-06-21-python-truthiness-vs-java-boolean.md](2026-06-21-python-truthiness-vs-java-boolean.md) |
| 平台 import 问题：先找导入链 | [2026-06-19-python-package-import-and-triton.md](2026-06-19-python-package-import-and-triton.md) |
| TokenizerManager 与 DetokenizerManager | [2026-06-19-tokenizer-vs-detokenizer-manager.md](2026-06-19-tokenizer-vs-detokenizer-manager.md) |
| Chat 请求：messages 先变成模型输入 | [2026-06-18-sglang-chat-request-entry.md](2026-06-18-sglang-chat-request-entry.md) |
| 当前 Server 启动：先解析，再装配进程 | [2026-06-18-sglang-server-launch-flow.md](2026-06-18-sglang-server-launch-flow.md) |
| Prefill、EXTEND、Chunk 与 Decode | [2026-06-19-prefill-extend-chunked-prefill-decode.md](2026-06-19-prefill-extend-chunked-prefill-decode.md) |


真正的运行证据集中在 [Mac 验证记录](../setup/mac-validation.md)，架构变更集中在 [当前代码地图](../01-architecture/current-code-map.md)。
