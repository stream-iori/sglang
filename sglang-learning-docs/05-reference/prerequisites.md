# 前置知识：Python 构建 + Transformers + 数学基础

> 目标：花 2-3 小时补齐看 SGLang 源码所需的背景知识。
> 受众：会写 Python 脚本，但没接触过 ML/深度学习的开发者。
> 方式：每个概念附带**可直接运行的代码**，不讲理论细节。
>
> **下一步**: 完成本文后，开始 [Week 1](../01-architecture/foundations.md) 或先搭建 [Mac 调试环境](../setup/mac-debug.md)。
> **进阶数学**: 如需了解 GPU 算力/带宽对性能的影响，见 [performance-intuition.md](../05-reference/performance-intuition.md)。

---

## Part 1: Python 构建基础

### 1.1 虚拟环境 (venv / uv)

**为什么不直接 `pip install`？**

直接装在系统 Python 里，不同项目的依赖会冲突（A 要 torch 2.0，B 要 torch 2.3）。虚拟环境 = 每个项目有自己独立的 `site-packages`。

```bash
# 创建虚拟环境 (uv 是更快的替代品，用法几乎和 venv 一样)
uv venv python/.venv --python 3.13

# 激活后，python/pip 指向虚拟环境里的版本
source python/.venv/bin/activate
which python  # → .../python/.venv/bin/python

# 安装包只影响这个虚拟环境
pip install numpy
```

### 1.2 PYTHONPATH 的作用

Python 在 `import` 时会在一系列目录中搜索模块。`PYTHONPATH` 可以往这个搜索列表里加目录。

```bash
# SGLang 的源码在 python/ 目录下，兼容层在 sglang-learning-docs/ 下
# 不设置 PYTHONPATH 的话，Python 找不到它们
PYTHONPATH="python" python -c "import sglang; print('找到了!')"
```

**为什么 SGLang 要这样做？** 因为 SGLang 用的是"开发模式"——直接从源码目录 import，而不是先 `pip install` 再用。这样你改了代码立刻生效，不用重新安装。

### 1.3 import 机制

```python
# Python 搜索模块的顺序:
import sys
print(sys.path)
# 1. 当前目录
# 2. PYTHONPATH 里的目录
# 3. 虚拟环境的 site-packages
# 4. 系统 Python 的标准库

# 包 vs 模块:
# sglang/srt/managers/scheduler.py  ← 这是一个模块
# sglang/srt/managers/               ← 这是一个包 (有 __init__.py)
# import sglang.srt.managers.scheduler  ← 完整路径导入
```

### 1.4 给 Java 开发者的 Python 速记

如果你有 Java 背景，下表帮你快速建立对应关系：

| Python | Java 对应 | 说明 |
|---|---|---|
| `self` | `this` | Python 要求显式写在方法第一个参数，Java 隐式存在 |
| `**kwargs` | `Map<String, Object>` | 接收任意关键字参数；`Foo(**kwargs)` ≈ 用 map 的值填构造函数参数 |
| `@dataclass` | `record` (Java 14+) 或 Lombok `@Data` | 自动生成 `__init__`、`__eq__`、`__repr__` |
| `with open(...) as f:` | try-with-resources | 自动关闭资源，Python 叫 context manager |
| 多继承 + Mixin | 多个带 `default` 方法的 interface | 但 Python Mixin 可带字段（状态），Java interface 不行 |
| `ABC` + `@abstractmethod` | `abstract class` / `interface` | Python 无 `interface` 关键字，用 `abc` 模块模拟 |
| `list[int]` / `dict[str, Any]` | `List<Integer>` / `Map<String, Object>` | 类型注解不影响运行，仅供阅读 and IDE 补全 |
| `lambda x: x + 1` | `x -> x + 1` | 匿名函数 |
| 装饰器 `@foo` | 注解 `@Foo` + AOP | 装饰器在运行时包装函数，比 Java 注解更动态 |

```python
# Mixin 示例 — Java 开发者容易困惑的多继承
class LogMixin:
    """类似 Java 中一个带 default 方法的 interface"""
    def log(self, msg):
        print(f"[{self.__class__.__name__}] {msg}")

class Scheduler(LogMixin, SomeMixin):
    """相当于 class Scheduler implements LogMixin, SomeMixin"""
    def run(self):
        self.log("running")  # 从 LogMixin 继承来的方法

# SGLang 中看到 class Scheduler(MixinA, MixinB, MixinC):
# 只需关注 Scheduler 自己的方法，Mixin 按需深入
```

> **Tips**: 遇到不认识的 Python 语法，搜 "Python xxx for Java developers" 通常能找到很好的对比文章。

### 1.5 常见 Python Pattern (SGLang 中大量使用)

```python
# 1. dataclass — 自动生成 __init__ 的数据容器
from dataclasses import dataclass

@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 30000
    model_path: str = ""

config = ServerConfig(model_path="meta-llama/Llama-3-8B")
print(config.port)  # 30000

# 2. Type Hints — 标注类型，不影响运行，但帮助阅读和 IDE 补全
def tokenize(text: str) -> list[int]:
    return [ord(c) for c in text]  # 简化示意

# 3. Enum — 枚举类型，比字符串常量更安全
from enum import IntEnum, auto
class ForwardMode(IntEnum):
    EXTEND = auto()   # 1
    DECODE = auto()   # 2
    MIXED = auto()    # 3

# SGLang 用 ForwardMode 区分 prefill 和 decode 阶段
```

### 1.6 多进程与 GIL 限制

SGLang 用多进程（不是多线程）来并行处理。为什么？

> **🎤 GIL (全局解释器锁)：单麦克风会议室比喻**
> 想象一间会议室（Python 解释器进程）里有 4 个歌唱家（线程）。虽然每个人都可以唱不同的歌，但**房间里只有一个麦克风**（GIL）。在任何时刻，只能有一个歌唱家拿着麦克风唱歌（执行 Python 字节码）。其他 3 人只能干等。
> 
> 既然多线程在 Python 里不能真正同时利用多核 CPU，该怎么办？
> **答案是：多进程！** 每一个歌唱家分一间独立的会议室（不同的 CPU 核心和独立的内存空间），各自拿自己的麦克风唱歌。由于房间完全隔离，数据互不干扰，他们就能真正同时唱了。

```python
from multiprocessing import Process, Queue
import time

def worker(name, q):
    """子进程：从队列取任务，处理后放回结果"""
    while True:
        task = q.get()
        if task == "STOP":
            break
        print(f"[{name}] 处理: {task}")
        time.sleep(0.1)

# 主进程创建队列和子进程
q = Queue()
p = Process(target=worker, args=("Worker-1", q))
p.start()

q.put("任务A")
q.put("任务B")
q.put("STOP")
p.join()
# SGLang 的 ZMQ 本质上做的事和 Queue 一样，但更灵活、跨机器
```

### 1.7 异步编程 (asyncio & async/await)

SGLang 的 HTTP 服务器（FastAPI）和 Tokenizer 管理器大量使用了异步编程。大一的同学可能只接触过同步（Synchronous）代码，遇到 `async` / `await` 时会很困惑。

> **☕️ 核心概念：茶餐厅服务员比喻**
> 想象你开了一家茶餐厅（CPU）：
> * **同步方式 (Synchronous)**：服务员走到 A 桌递上菜单，**一直站在桌边等**客人点完菜（网络 I/O 阻塞），把菜单送回厨房，做好了再端上桌。然后才能去服务 B 桌。如果 A 桌客人看菜单看了一个小时，服务员就被“阻塞”了一个小时，茶餐厅就破产了。
> * **异步方式 (Asynchronous)**：服务员把菜单给 A 桌，说“点好了叫我”（`await`），接着立刻去服务 B 桌。当 A 桌点好了喊一声“服务员！”（触发事件），服务员就跑回去处理。服务员（CPU）一刻不停地在运转，没有闲置。

**💻 动手试一试**：你可以在 Mac 上直接新建运行如下 Python 脚本，观察同步与异步 3 次“点菜”任务的时耗差距：

```python
import asyncio
import time

# 同步函数：服务员傻等
def sync_order(table):
    print(f"[同步] 开始服务 {table} 桌...")
    time.sleep(1) # 模拟等客人看菜单 1 秒
    print(f"[同步] {table} 桌点菜完毕！")

# 异步函数：服务员去干别的
async def async_order(table):
    print(f"[异步] 开始服务 {table} 桌...")
    await asyncio.sleep(1) # 挂起，把控制权交还给事件循环
    print(f"[异步] {table} 桌点菜完毕！")

def run_demo():
    # 1. 跑同步版本
    print("=== 开始运行同步版本 ===")
    start = time.time()
    for table in ["A", "B", "C"]:
        sync_order(table)
    print(f"同步总共耗时: {time.time() - start:.2f} 秒\n")

    # 2. 跑异步版本
    print("=== 开始运行异步版本 ===")
    start = time.time()
    async def main():
        # 并发跑三个异步任务
        await asyncio.gather(
            async_order("A"),
            async_order("B"),
            async_order("C")
        )
    asyncio.run(main())
    print(f"异步总共耗时: {time.time() - start:.2f} 秒")

if __name__ == "__main__":
    run_demo()
```
运行后你会发现，同步用了 **3.00 秒**，而异步只用了 **1.00 秒**！这就是为什么 SGLang 能并发处理数千个 HTTP 请求而不会卡住。

### 1.8 Python 中的对象引用 (值传递 vs 引用传递)

在 Week 2 的 `RadixCache` 中，你会遇到引用计数 `lock_ref` 的概念。如果大一同学对 Python 的对象存储机制不清晰，可能会在这个概念上栽跟头。

**💡 核心要点：Python 中的变量全是“指针”**
与 C/C++ 不同，在 Python 中：
1. `a = [1, 2, 3]`：是在内存中创建了一个列表对象，并让名字 `a` 指向它（也就是存了它的内存地址/引用）。
2. `b = a`：**并没有复制这个列表**！它只是把 `b` 这个名字也指向同一个列表对象的内存地址。
3. `b.append(4)` 会导致 `a` 也变成 `[1, 2, 3, 4]`，因为它们操作的是同一个内存块。

**📌 与 RadixCache 的关联**：
当 Scheduler 将一个请求（`Req`）挂载 to RadixCache 树的某个节点（`TreeNode`）时，`TreeNode` 的 `lock_ref` 计数会加 1。只要有活跃请求的变量依然指向这个节点，这个节点就处于“锁定”状态，绝对不会被 LRU 驱逐。只有等请求处理完毕并断开引用后，`lock_ref` 降为 0，该节点的 KV 显存才能被安全释放或复用。

---

## Part 2: Transformers 快速理解

### 2.1 LLM 是什么？

大语言模型 (LLM) 本质上就是一个函数：

```
输入: 一段文字 (如 "今天天气")
输出: 下一个字的概率分布 (如 {"很": 0.6, "不": 0.2, "还": 0.1, ...})
```

然后从概率分布中选一个字（采样），拼到输入后面，再重复。这就是"自回归生成"。

### 2.2 Tokenizer：文字 ↔ 数字

模型不认识文字，只认识数字。Tokenizer 负责转换。

```python
# 可在 Mac 上直接运行 (需先 pip install transformers)
from transformers import AutoTokenizer

# 加载一个 tokenizer (不需要 GPU，只是查表操作)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 文字 → 数字 (encode)
text = "Hello, world!"
ids = tokenizer.encode(text)
print(f"文字: {text}")
print(f"Token IDs: {ids}")        # [15496, 11, 995, 0]
print(f"Token 数量: {len(ids)}")  # 4

# 数字 → 文字 (decode)
recovered = tokenizer.decode(ids)
print(f"恢复: {recovered}")       # "Hello, world!"

# 查看每个 token 对应什么
for id in ids:
    print(f"  {id} → '{tokenizer.decode([id])}'")
```

**关键理解**: 一个 token ≠ 一个字。英文中一个 token 约 4 个字母，中文中一个 token 约 1-2 个字。

更完整的 tokenizer 机制，包括 vocab、BPE/SentencePiece、special token、chat template 和 SGLang 的 TokenizerManager/DetokenizerManager，见 [Tokenizer 内部机制](./tokenizer-internals.md)。

### 2.3 Chat Template (对话模板)

ChatGPT 风格的对话有固定格式：

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi! How can I help?"},
    {"role": "user", "content": "What is 2+2?"},
]

# Chat template 把上面的结构转成模型能理解 of 纯文本:
# <|system|>You are a helpful assistant.<|end|>
# <|user|>Hello!<|end|>
# <|assistant|>Hi! How can I help?<|end|>
# <|user|>What is 2+2?<|end|>
# <|assistant|>

# SGLang 中 TokenizerManager 负责这个转换
```

<a id="transformer-basics-sglang"></a>

### 2.4 Transformer 架构与 SGLang 映射

为了使前置知识结构更清晰，我们已将 Transformer 相关的全部理论、架构原理、Decoder-Only 演进、极简 Python 自回归仿真、Multi-Head 维度拆分、RMSNorm/RoPE/GQA/SwiGLU 等微观架构以及 SGLang 的对应文件映射独立拆分为一门专门的参考课：

👉 **[现代大模型 Transformer 架构：从原理到 SGLang 映射](./transformer.md)**

建议完成 Tokenizer 和对话模板的学习后，点击上方链接阅读该核心架构说明，然后再返回阅读后续的数学基础部分。

---

## Part 3: 数学基础

数学内容已经拆成独立文档：[LLM 推理数学基础](./math-for-llm.md)。

这篇文档按大一新生水平讲：

| 主题 | 为什么要学 |
|---|---|
| 标量/向量/矩阵/张量 | 看懂 `input_ids`, `hidden_states`, `logits` 的 shape |
| 矩阵乘法 | 看懂 Transformer 每层在做什么 |
| logits/softmax/概率 | 看懂 sampler 为什么能选 token |
| temperature/top-p | 看懂采样参数如何影响输出 |
| Attention/QKV | 看懂 Transformer 为什么能读上下文 |
| KV Cache | 看懂 SGLang 为什么重视显存管理和前缀缓存 |

如果你没有线性代数、概率论基础，先读它，再进入 Week 1。

---

## 自查：你准备好了吗？

读完上面的内容后，你应该能回答：

- [ ] `PYTHONPATH="python"` 这行命令是在做什么？
- [ ] `tokenizer.encode("Hello world")` 的返回值是什么类型？
- [ ] `token_id` 为什么要转成 embedding？
- [ ] logits 和 probability 有什么区别？
- [ ] KV Cache 为什么能加速 decode？
- [ ] 为什么 SGLang 用多进程而不是多线程？(GIL)

全部能答上来 → 你已经准备好进入 Week 1 了！
