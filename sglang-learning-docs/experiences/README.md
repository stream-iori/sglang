# 读码补充笔记

> 源码基准：上游 `16eb87c06f`；本地集成 `4f0e1cbea2`（2026-10-20）。本地日志增强与验证范围见总索引和 Mac 验证记录。

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

## 阅读顺序与验收

先读引用、签名、self/cls、import/PYTHONPATH，再读启动、chat、prefill/decode 和增量解码。每篇都有可运行小例子或读码练习；语法实验解释语言行为，真实 SRT 行为仍回到对应构造与调用分支验证。

阶段验收：用自己的话解释“两个变量指向同一 Req”“两个进程各有配置”“两个 runner 的地址视角不同”。三者属于不同层次，不能用同一种共享对象类比覆盖。
