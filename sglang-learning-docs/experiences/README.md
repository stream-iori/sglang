# Experiences —— 踩坑 & 知识点随手记

> 学习 SGLang 源码过程中遇到的零散知识点、踩的坑、看不懂的命令/语法，随手记在这里。
> 区别于 `05-reference/`（成体系的参考手册），这里是**按时间、按问题**的流水账，更口语、更碎片。

## 写法约定

- 文件名：`YYYY-MM-DD-<英文短描述>.md`，一篇记一个问题。
- 开头固定三行：`记录日期` / `触发场景` / `一句话结论`。
- 多用 Mermaid 图和 ASCII 图、表格，面向初学者，能看懂比严谨更重要。

## 索引

> 新增笔记后，在这里补一行（最新的放最上面）。

| 日期 | 笔记 | 一句话 |
|------|------|--------|
| 2026-06-21 | [Python Truthiness 对比 Java Boolean](2026-06-21-python-truthiness-vs-java-boolean.md) | `if x` 按 `__bool__`→`__len__`→对象默认真的顺序判断；空 `ScheduleBatch` 仍为真 |
| 2026-06-21 | [Python `self` / `cls` 对比 Java `this` / 工厂模式](2026-06-21-python-self-cls-vs-java-this-factory.md) | `self` 是显式的 `this`；`cls` 支持多态构造，更接近 Factory Method 而非 Abstract Factory |
| 2026-06-19 | [Prefill / Extend / Chunked Prefill / Decode](2026-06-19-prefill-extend-chunked-prefill-decode.md) | extend 是调度层追加 token; prefill 是建 KV cache; chunked prefill 是长 prompt 多次 extend; decode 是逐 token 生成 |
| 2026-06-19 | [Python 包导入链 & Triton 被意外加载](2026-06-19-python-package-import-and-triton.md) | 导入子模块前会先执行父包 `__init__.py`;轻量工具放进重包可能被父包副作用带去 import Triton |
| 2026-06-19 | [TokenizerManager vs DetokenizerManager 区别](2026-06-19-tokenizer-vs-detokenizer-manager.md) | TokenizerManager 负责接收请求与 Token 编码，DetokenizerManager 负责生成 Token ID 的解码与流式安全后处理 |
| 2026-06-18 | [Chat 请求入口链路](2026-06-18-sglang-chat-request-entry.md) | /v1/chat/completions → 路由薄壳 → handle_request 模板方法 → tokenizer_manager.generate_request 入引擎 |
| 2026-06-18 | [如何找 ABC 抽象方法的实现](2026-06-18-find-abstractmethod-implementation.md) | 实现在子类；`grep "def 方法名"` 或 IDE「Go to Implementations」最快，多态决定运行时调哪个 |
| 2026-06-18 | [SGLang Server 启动全链路](2026-06-18-sglang-server-launch-flow.md) | launch_server = 起子进程拿对象(_launch_subprocesses) + 装配跑 HTTP(_setup_and_run_http_server) + warmup |
| 2026-06-18 | [局部 import(lazy import)](2026-06-18-python-local-import.md) | import 写进函数=延迟到调用时加载，为避开可选重依赖(Ray)和循环 import |
| 2026-06-18 | [Python 函数签名语法 vs Java](2026-06-18-python-function-signature-vs-java.md) | 类型注解/Callable/Optional/函数当默认值=依赖注入；Java 要靠函数式接口+方法引用+重载 |
| 2026-06-18 | [SGLang 插件机制 & entry_points](2026-06-18-sglang-plugins-and-entry-points.md) | entry_points 是「全局报名表」，pip 装了插件包 SGLang 启动就自动发现，主仓库零改动 |
| 2026-06-18 | [Python 对象引用 vs Java](2026-06-18-python-vs-java-object-reference.md) | 引用语义像 Java，但有三个坑：无基本类型 / 可变性 / `is`和`==`符号对调 |
| 2026-06-18 | [Python 的两个 `__init__`](2026-06-18-python-init-explained.md) | `__init__.py`（包标志/门面）vs `def __init__(self)`（对象初始化器），名字像但无关 |
| 2026-06-18 | [PYTHONPATH 是什么](2026-06-18-pythonpath-explained.md) | 临时给 Python 加「找模块的目录」，让你不安装就能直接用仓库源码跑脚本 |

## 主题速查

- **Python 语言基础**: [Truthiness vs Java Boolean](2026-06-21-python-truthiness-vs-java-boolean.md) · [`self` / `cls` vs Java `this` / 工厂](2026-06-21-python-self-cls-vs-java-this-factory.md) · [`__init__`](2026-06-18-python-init-explained.md) · [对象引用 vs Java](2026-06-18-python-vs-java-object-reference.md) · [函数签名语法 vs Java](2026-06-18-python-function-signature-vs-java.md) · [局部 import](2026-06-18-python-local-import.md) · [包导入链 & Triton](2026-06-19-python-package-import-and-triton.md)
- **看码技能**: [找 ABC 抽象方法的实现](2026-06-18-find-abstractmethod-implementation.md)
- **环境 / 运行命令**: [PYTHONPATH](2026-06-18-pythonpath-explained.md) · [包导入链 & Triton](2026-06-19-python-package-import-and-triton.md)
- **SGLang 内部机制**: [Prefill / Extend / Chunked Prefill / Decode](2026-06-19-prefill-extend-chunked-prefill-decode.md) · [TokenizerManager vs DetokenizerManager](2026-06-19-tokenizer-vs-detokenizer-manager.md) · [插件 & entry_points](2026-06-18-sglang-plugins-and-entry-points.md) · [Server 启动全链路](2026-06-18-sglang-server-launch-flow.md) · [Chat 请求入口链路](2026-06-18-sglang-chat-request-entry.md)
