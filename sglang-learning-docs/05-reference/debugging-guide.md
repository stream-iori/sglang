# 调试实战指南

> 目标：掌握在 Mac (CPU) 和远程 GPU 环境中调试 SGLang 源码的实用技巧。
> 前置：[Week 1](../01-architecture/foundations.md) 完成，环境已搭建（[mac-debug.md](../setup/mac-debug.md)）。
> 时间：~2 小时 (按需查阅)
>
> **练手建议**: 配合 [exercises.md](../04-practice/exercises.md) 的 demo 脚本练习断点调试，比直接调试 SGLang 源码更友好。

---

## 一、Python 断点调试 (pdb / breakpoint)

### 1.1 最简单的方式: breakpoint()

在任何想暂停的位置插入一行：

```python
# 例如想看 Scheduler 收到请求后的状态
# 文件: python/sglang/srt/managers/scheduler.py

def process_input_requests(self, recv_reqs):
    for recv_req in recv_reqs:
        breakpoint()  # ← 在这里暂停
        # 暂停后你可以:
        # p recv_req          — 打印请求内容
        # p self.waiting_queue — 查看等待队列
        # n                   — 执行下一行
        # c                   — 继续运行
        # bt                  — 打印调用栈
```

### 1.2 条件断点

```python
# 只在特定条件下暂停 (避免被大量请求打断)
if recv_req.rid == "target-request-id":
    breakpoint()
```

### 1.3 单测中使用断点

```bash
# pytest 默认会捕获 stdout，需要加 -s 参数
PYTHONPATH="python" python -m pytest \
  test/registered/unit/mem_cache/test_radix_cache_unit.py \
  -s -v -k "test_insert"

# 在测试代码或被测代码中插入 breakpoint() 即可
```

### 1.4 pdb 常用命令速查

| 命令 | 缩写 | 作用 |
|---|---|---|
| `next` | `n` | 执行下一行 (不进入函数) |
| `step` | `s` | 进入函数内部 |
| `continue` | `c` | 继续运行到下一个断点 |
| `print expr` | `p expr` | 打印表达式的值 |
| `pp expr` | | 美化打印 (适合大对象) |
| `where` | `bt` | 打印完整调用栈 |
| `up` / `down` | `u` / `d` | 在调用栈中上下移动 |
| `list` | `l` | 显示当前代码上下文 |
| `args` | `a` | 打印当前函数的参数 |
| `quit` | `q` | 退出调试 |

---

## 二、VS Code 图形化调试

### 2.1 配置 launch.json

创建 `.vscode/launch.json`:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Debug Unit Test",
      "type": "debugpy",
      "request": "launch",
      "module": "pytest",
      "args": [
        "test/registered/unit/mem_cache/test_radix_cache_unit.py",
        "-v", "-s", "-k", "test_insert"
      ],
      "env": {
        "PYTHONPATH": "${workspaceFolder}/python"
      },
      "cwd": "${workspaceFolder}",
      "justMyCode": false
    },
    {
      "name": "Debug Demo Script",
      "type": "debugpy",
      "request": "launch",
      "program": "${workspaceFolder}/sglang-learning-docs/06_demo_scheduler.py",
      "env": {
        "PYTHONPATH": "${workspaceFolder}/python"
      },
      "cwd": "${workspaceFolder}",
      "justMyCode": false
    }
  ]
}
```

### 2.2 调试操作

1. 在代码行号左侧点击 → 设置红色断点
2. 按 F5 启动调试
3. 程序暂停后:
   - **Variables 面板**: 查看所有局部变量
   - **Watch 面板**: 添加自定义表达式监控
   - **Call Stack 面板**: 查看完整调用链
   - **Debug Console**: 在当前上下文执行任意 Python 表达式

### 2.3 调试多进程

SGLang 使用多进程架构，默认调试器只附加到主进程。

方法 1: 调试子进程 (attach)
```json
{
  "name": "Attach to Scheduler",
  "type": "debugpy",
  "request": "attach",
  "connect": {"host": "localhost", "port": 5678}
}
```

在 scheduler 启动代码中加入:
```python
import debugpy
debugpy.listen(5678)
debugpy.wait_for_client()  # 等待 VS Code 连接
```

方法 2: 使用单进程模式 (推荐学习时用)
```python
# 直接实例化 Scheduler 类进行单元测试，绕过多进程
from sglang.srt.managers.scheduler import Scheduler
```

---

## 三、日志系统

### 3.1 SGLang 内置日志级别

```bash
# 启动 server 时设置日志级别
python -m sglang.launch_server --model ... --log-level debug

# 可选级别: debug, info, warning, error
# debug 会打印大量调度细节 (batch 组成、缓存命中等)
```

### 3.2 自定义日志配置

```bash
# 使用自定义日志配置文件
export SGLANG_LOGGING_CONFIG_PATH=my_logging.json
```

`my_logging.json` 示例:
```json
{
  "version": 1,
  "disable_existing_loggers": false,
  "formatters": {
    "detailed": {
      "format": "[%(asctime)s] %(name)s:%(lineno)d %(levelname)s: %(message)s"
    }
  },
  "handlers": {
    "file": {
      "class": "logging.FileHandler",
      "filename": "sglang_debug.log",
      "formatter": "detailed"
    },
    "console": {
      "class": "logging.StreamHandler",
      "formatter": "detailed"
    }
  },
  "root": {
    "level": "DEBUG",
    "handlers": ["console", "file"]
  }
}
```

### 3.3 在源码中临时添加日志

```python
import logging
logger = logging.getLogger(__name__)

# 在关键位置添加
logger.debug(f"get_next_batch: waiting={len(self.waiting_queue)}, "
             f"running={len(self.running_batch.reqs)}")
```

### 3.4 有用的环境变量

| 环境变量 | 作用 |
|---|---|
| `--log-level debug` | 打开 debug 级别日志 |
| `SGLANG_LOGGING_CONFIG_PATH` | 自定义日志配置文件路径 |
| `SGLANG_DEBUG_MEMORY_POOL=1` | 内存池调试信息 |
| `SGLANG_LOG_MS=1` | 日志显示毫秒级时间戳 |

---

## 四、性能分析: py-spy 火焰图

### 4.1 安装

```bash
pip install py-spy
```

### 4.2 对运行中的进程采样

```bash
# 1. 找到 SGLang 进程 PID
ps aux | grep sglang

# 2. 生成火焰图 (采样 30 秒)
sudo py-spy record -o flamegraph.svg --pid <PID> --duration 30

# 3. 用浏览器打开 SVG 文件
open flamegraph.svg
```

### 4.3 对单测生成火焰图

```bash
py-spy record -o test_flame.svg -- python -m pytest \
  test/registered/unit/mem_cache/test_radix_cache_unit.py -v
```

### 4.4 如何解读火焰图

```
┌──────────────────────────────────────────────────┐
│                    main()                          │  ← 最上层: 入口
├────────────────────────┬─────────────────────────┤
│   event_loop_normal()  │   recv_requests()        │  ← 两个主要分支
├──────────┬─────────────┤                         │
│run_batch │get_next_batch│                         │
├──────────┤             │                         │
│ forward()│             │                         │  ← 越底层越具体
└──────────┴─────────────┴─────────────────────────┘

宽度 = 该函数占用的 CPU 时间比例
越宽 = 越热 (可能是瓶颈)
```

**看什么**:
- 最宽的"平顶"区域 = CPU 热点
- 出乎意料的宽条 = 可能有性能 bug
- 对比 prefill 和 decode 时的火焰图差异

---

## 五、常见报错排查

### 5.1 ImportError: No module named 'xxx'

```bash
# 症状
ImportError: No module named 'sglang'
ImportError: No module named 'triton'

# 原因: PYTHONPATH 未设置
# 解决:
export PYTHONPATH="python"

# 验证:
python -c "import sglang; print(sglang.__file__)"
```

### 5.2 ZMQ Address Already in Use

```bash
# 症状
zmq.error.ZMQError: Address already in use (addr='tcp://127.0.0.1:30000')

# 原因: 上次运行没有正常退出，端口被占用
# 解决:
lsof -i :30000  # 找到占用进程
kill <PID>      # 杀掉它

# 或者换端口:
python -m sglang.launch_server --port 30001 ...
```

### 5.3 CUDA Out of Memory (远程 GPU)

```bash
# 症状
torch.cuda.OutOfMemoryError: CUDA out of memory

# 排查步骤:
# 1. 查看 GPU 显存使用
nvidia-smi

# 2. 检查是否有残余进程
nvidia-smi | grep python
kill <PID>

# 3. 减小 batch size 或 max_total_tokens
python -m sglang.launch_server \
  --model ... \
  --mem-fraction-static 0.8  # 降低显存使用比例 (默认 0.88)
```

### 5.4 请求超时 / 卡住

```bash
# 症状: 请求发出去但没有响应

# 排查步骤:
# 1. 检查各进程是否活着
ps aux | grep sglang

# 2. 看日志中是否有 ERROR
grep -i "error\|exception\|traceback" sglang_*.log

# 3. 用 py-spy 查看卡在哪里
py-spy dump --pid <scheduler_pid>
# 输出会显示每个线程的当前调用栈

# 4. 常见原因:
#    - Scheduler 在等 GPU forward (正常，但如果太久则可能死锁)
#    - ZMQ recv() 阻塞 (上游进程可能已崩)
#    - 内存不足导致请求一直在 waiting_queue 等待
```

### 5.5 结果不正确 / 乱码

```bash
# 可能原因:
# 1. tokenizer 版本不匹配 (模型需要特定版本的 tokenizer)
# 2. chat template 应用错误

# 调试: 打印中间结果
# 在 tokenizer_manager.py 中:
logger.debug(f"tokens: {input_ids[:20]}...")
logger.debug(f"decoded: {tokenizer.decode(input_ids[:20])}")
```

---

## 六、实用调试技巧

### 6.1 快速定位函数

```bash
# 在 SGLang 代码中搜索关键函数
grep -rn "def get_next_batch_to_run" python/sglang/srt/
grep -rn "def forward" python/sglang/srt/model_executor/

# 找某个类的所有方法
grep -n "def " python/sglang/srt/managers/scheduler.py | head -30
```

### 6.2 打印对象结构

```python
# 在断点中查看复杂对象
import pprint

# 查看 Req 对象的所有属性
p vars(req)

# 查看 ScheduleBatch 的结构
pp {k: type(v).__name__ for k, v in vars(batch).items()}
```

### 6.3 追踪函数调用

```python
import traceback

# 打印"谁调用了我"
def some_function(self, ...):
    traceback.print_stack()  # 打印完整调用链
    ...
```

### 6.4 计时某段代码

```python
import time

start = time.perf_counter()
# ... 要计时的代码 ...
elapsed = time.perf_counter() - start
logger.info(f"forward took {elapsed*1000:.1f}ms")
```

---

## 七、调试工作流推荐

### 学习阶段 (Mac CPU)

```
1. 选一个单测作为入口
2. 在感兴趣的函数设断点
3. 用 VS Code 调试运行
4. 在 Variables 面板观察数据流
5. 用 Step Into 追踪调用链
```

### 排查 Bug (远程 GPU)

```
1. 复现问题 (构造最小请求)
2. 加 --log-level debug 看日志
3. 日志定位到大致位置后，加 breakpoint()
4. 如果是性能问题，用 py-spy 采样
5. 如果是死锁，用 py-spy dump 看调用栈
```

### 理解数据流

```
1. 在 tokenizer_manager 的入口打印 input
2. 在 scheduler 的入口打印 tokenized input
3. 在 model_runner 的入口打印 ForwardBatch
4. 在 detokenizer 的入口打印 output tokens
5. 对照上述 4 个打印，理解数据如何变换
```
