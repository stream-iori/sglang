# 动手实验：可运行的渐进式 Demo

> 目标：通过 3 个独立的可运行 demo，亲手体验 SGLang 的核心机制。
> 环境：Mac CPU，无需 GPU。所有代码可直接复制运行。
> 前置：`setup_mac.sh` 已执行完成（见 [mac-debug.md](../setup/mac-debug.md)）。
>
> **配合阅读**: 实验 1 对应 [Week 2 Scheduler 章节](../02-core-systems/scheduler-and-cache.md)；实验 2 对应 [Week 2 RadixCache 章节](../02-core-systems/scheduler-and-cache.md#day-3-4-radixcache-前缀缓存-4h)；实验 3 对应 [Week 1 多进程架构](../01-architecture/foundations.md)。
> **调试技巧**: 如果想在 demo 中设断点，参考 [debugging-guide.md](../05-reference/debugging-guide.md) 的 pdb 用法。
> **做完对答案**: 看 [exercise-solutions.md](./exercise-solutions.md)。

---

## 实验 1: Scheduler 主循环模拟

这个 demo 模拟了 `event_loop_normal()` 的核心逻辑：收请求 → 调度 → 执行 → 返回结果。

```python
"""
模拟 SGLang Scheduler 主循环。
对应源码: python/sglang/srt/managers/scheduler.py:event_loop_normal()

运行: python sglang-learning-docs/06_demo_scheduler.py
"""

import time
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class ForwardMode(Enum):
    EXTEND = "extend"   # Prefill: 首次处理完整 prompt
    DECODE = "decode"   # Decode: 逐 token 生成


@dataclass
class Req:
    rid: str
    prompt_tokens: List[int]
    output_tokens: List[int] = field(default_factory=list)
    max_new_tokens: int = 10
    finished: bool = False

    @property
    def is_prefilled(self) -> bool:
        return len(self.output_tokens) > 0


@dataclass
class ScheduleBatch:
    reqs: List[Req]
    forward_mode: ForwardMode


class FakeModel:
    """模拟模型前向，返回随机 token"""
    def forward(self, batch: ScheduleBatch) -> List[int]:
        time.sleep(0.05)  # 模拟计算延迟
        return [random.randint(100, 999) for _ in batch.reqs]


class MiniScheduler:
    def __init__(self):
        self.waiting_queue: List[Req] = []
        self.running_batch: List[Req] = []
        self.model = FakeModel()
        self.finished: List[Req] = []

    def add_request(self, req: Req):
        self.waiting_queue.append(req)
        print(f"  [收到请求] rid={req.rid}, prompt_len={len(req.prompt_tokens)}")

    def get_next_batch_to_run(self) -> Optional[ScheduleBatch]:
        # 1. 优先处理等待队列中的新请求 (prefill)
        if self.waiting_queue:
            new_reqs = self.waiting_queue[:4]  # 最多 batch_size=4
            self.waiting_queue = self.waiting_queue[4:]
            self.running_batch.extend(new_reqs)
            return ScheduleBatch(reqs=new_reqs, forward_mode=ForwardMode.EXTEND)

        # 2. 处理已在运行中的请求 (decode)
        if self.running_batch:
            return ScheduleBatch(reqs=self.running_batch, forward_mode=ForwardMode.DECODE)

        return None

    def run_batch(self, batch: ScheduleBatch) -> List[int]:
        mode_str = "PREFILL" if batch.forward_mode == ForwardMode.EXTEND else "DECODE"
        print(f"  [执行] mode={mode_str}, batch_size={len(batch.reqs)}")
        return self.model.forward(batch)

    def process_batch_result(self, batch: ScheduleBatch, next_tokens: List[int]):
        for req, token in zip(batch.reqs, next_tokens):
            req.output_tokens.append(token)
            if len(req.output_tokens) >= req.max_new_tokens:
                req.finished = True
                self.running_batch.remove(req)
                self.finished.append(req)
                print(f"  [完成] rid={req.rid}, output={req.output_tokens}")

    def event_loop_step(self):
        """对应 event_loop_normal() 的一次迭代"""
        batch = self.get_next_batch_to_run()
        if batch:
            result = self.run_batch(batch)
            self.process_batch_result(batch, result)
        else:
            return False  # idle
        return True


def main():
    scheduler = MiniScheduler()

    # 模拟请求陆续到达
    print("=" * 60)
    print("SGLang Scheduler 模拟")
    print("=" * 60)

    # 第 1 轮: 添加 2 个请求
    print("\n--- 添加 2 个请求 ---")
    scheduler.add_request(Req(rid="req-1", prompt_tokens=[1, 2, 3, 4, 5], max_new_tokens=3))
    scheduler.add_request(Req(rid="req-2", prompt_tokens=[10, 20, 30], max_new_tokens=5))

    # 运行调度循环
    print("\n--- 开始调度循环 ---")
    step = 0
    while scheduler.waiting_queue or scheduler.running_batch:
        step += 1
        print(f"\n[Step {step}] waiting={len(scheduler.waiting_queue)}, "
              f"running={len(scheduler.running_batch)}")
        scheduler.event_loop_step()

    # 第 2 轮: 中途追加请求 (演示 continuous batching)
    print("\n\n--- 演示 Continuous Batching: 中途追加请求 ---")
    scheduler.add_request(Req(rid="req-3", prompt_tokens=[5, 6, 7], max_new_tokens=2))
    scheduler.add_request(Req(rid="req-4", prompt_tokens=[8, 9], max_new_tokens=4))

    # req-3 会先 prefill，然后和 req-4 一起 decode
    step = 0
    while scheduler.waiting_queue or scheduler.running_batch:
        step += 1
        print(f"\n[Step {step}] waiting={len(scheduler.waiting_queue)}, "
              f"running={len(scheduler.running_batch)}")
        scheduler.event_loop_step()

    print(f"\n\n{'=' * 60}")
    print(f"全部完成! 共处理 {len(scheduler.finished)} 个请求")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

### 运行方式

```bash
python sglang-learning-docs/06_demo_scheduler.py
```

### 观察要点

1. **两种 ForwardMode**: 新请求先 EXTEND (prefill)，之后全部 DECODE
2. **Continuous Batching**: req-3 完成后，req-4 继续 decode，不等待新 batch
3. **对比源码**: 打开 `python/sglang/srt/managers/scheduler.py:1425` 的 `event_loop_normal()`，结构几乎一致

---

## 实验 2: RadixCache 前缀缓存交互

这个 demo 让你亲手操作 SGLang 真实的 RadixCache，体验前缀匹配和缓存淘汰。

```python
"""
RadixCache 交互式 Demo。
直接使用 SGLang 源码中的 RadixCache 类。

运行: PYTHONPATH="sglang-learning-docs:python" python sglang-learning-docs/06_demo_radix_cache.py
"""

import sys
sys.path.insert(0, "sglang-learning-docs")
sys.path.insert(0, "python")

from sglang.srt.mem_cache.radix_cache import RadixCache


class FakeTokenPool:
    """模拟 TokenToKVPool，不需要真实 GPU"""
    def __init__(self, size: int):
        self.size = size
        self.used = set()
        self._next_id = 0

    def available_size(self):
        return self.size - len(self.used)

    def alloc(self, n: int):
        indices = list(range(self._next_id, self._next_id + n))
        self._next_id += n
        self.used.update(indices)
        return indices

    def free(self, indices):
        self.used -= set(indices)


def main():
    print("=" * 60)
    print("RadixCache 前缀缓存 Demo")
    print("=" * 60)

    # 创建一个简易的 RadixCache
    # 注意: 真实的 RadixCache 需要配合 MemoryPool，这里我们直接演示其树结构
    from sglang.srt.mem_cache.radix_cache import TreeNode

    # 手动构建一棵 Radix Tree 来演示原理
    print("\n--- 原理演示: 手动构建 Radix Tree ---")
    print()

    # 模拟 3 个请求共享前缀:
    # req1: "Hello, how are you" → tokens [1, 2, 3, 4, 5]
    # req2: "Hello, how is it"   → tokens [1, 2, 3, 6, 7]
    # req3: "Hello, how are you doing" → tokens [1, 2, 3, 4, 5, 8, 9]

    tree = {}  # 简化版 radix tree

    def insert(tokens, label):
        """插入一个 token 序列到树中"""
        print(f"  插入 {label}: tokens={tokens}")

        # 找到最长公共前缀
        node = tree
        depth = 0
        for t in tokens:
            if t not in node:
                node[t] = {}
            node = node[t]
            depth += 1

        print(f"    → 新增节点数: {len(tokens) - depth + len(tokens)}")
        # 实际 RadixCache 会分配 KV Cache 页给新节点

    def match_prefix(tokens, label):
        """匹配最长前缀"""
        node = tree
        matched = 0
        for t in tokens:
            if t in node:
                node = node[t]
                matched += 1
            else:
                break
        print(f"  匹配 {label}: tokens={tokens}")
        print(f"    → 命中前缀长度: {matched}/{len(tokens)}")
        print(f"    → 节省计算: 前 {matched} 个 token 无需重新 prefill!")
        return matched

    # 演示插入和前缀匹配
    print("\n[Step 1] 插入第一个请求的 KV Cache")
    insert([1, 2, 3, 4, 5], "req1='Hello, how are you'")

    print("\n[Step 2] 第二个请求来了，匹配前缀")
    hit = match_prefix([1, 2, 3, 6, 7], "req2='Hello, how is it'")
    print(f"    → 只需 prefill 后 {5 - hit} 个 token (节省 {hit/5*100:.0f}% 计算)")
    insert([1, 2, 3, 6, 7], "req2")

    print("\n[Step 3] 第三个请求，与 req1 完全匹配")
    hit = match_prefix([1, 2, 3, 4, 5, 8, 9], "req3='Hello, how are you doing'")
    print(f"    → 只需 prefill 后 {7 - hit} 个 token (节省 {hit/7*100:.0f}% 计算)")

    # 缓存淘汰演示
    print("\n\n--- 缓存淘汰 (LRU) ---")
    print()
    print("  假设 KV Cache 总容量只有 10 个 token 的空间")
    print("  当前已用: req1(5) + req2 独有(2) + req3 独有(2) = 9 个位置")
    print("  新请求 req4 需要 3 个新位置 → 超出容量!")
    print()
    print("  淘汰策略: 找到 lock_ref=0 (没有活跃请求引用) 且最久未使用的叶子节点")
    print("  假设 req1 已完成 → 其独有的 tokens [4,5] 的 lock_ref=0")
    print("  淘汰 [4,5] → 释放 2 个位置")
    print("  但 [1,2,3] 仍被 req2 引用 (lock_ref>0)，不会被淘汰!")

    # 性能影响
    print("\n\n--- 性能影响: 为什么前缀缓存这么重要？ ---")
    print()
    print("  场景: 多轮对话 (system prompt 相同)")
    print("  System prompt: 500 tokens")
    print("  每轮对话追加: ~50 tokens")
    print()
    print("  无缓存: 每轮都要 prefill 500+50+50+... tokens")
    print("  有缓存: 第 2 轮只需 prefill 50 tokens (节省 90%+)")
    print()
    print("  场景: Batch 请求 (相同 prompt 不同参数)")
    print("  共享前缀: 'Translate the following:' = 100 tokens")
    print("  100 个请求 → 节省 100×100 = 10000 tokens 的重复计算!")


if __name__ == "__main__":
    main()
```

### 运行方式

```bash
PYTHONPATH="sglang-learning-docs:python" python sglang-learning-docs/06_demo_radix_cache.py
```

### 观察要点

1. **前缀匹配**: 共享前缀越长，节省的 prefill 计算越多
2. **lock_ref**: 活跃请求"锁定"它使用的缓存节点，防止被淘汰
3. **对比源码**: `python/sglang/srt/mem_cache/radix_cache.py:333` 的 `match_prefix()`

---

## 实验 3: ZMQ 多进程通信流水线

这个 demo 模拟 SGLang 的多进程架构：HTTP → Tokenizer → Scheduler → Detokenizer。

```python
"""
ZMQ 多进程通信 Demo。
模拟 SGLang 的 4 进程流水线通信。

对应源码:
- TokenizerManager: python/sglang/srt/managers/tokenizer_manager.py
- Scheduler: python/sglang/srt/managers/scheduler.py
- DetokenizerManager: python/sglang/srt/managers/detokenizer_manager.py

运行: python sglang-learning-docs/06_demo_zmq_pipeline.py
"""

import multiprocessing
import time
import json
import zmq


def http_server(tokenizer_addr: str):
    """模拟 HTTP Server (FastAPI)"""
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUSH)
    sock.connect(tokenizer_addr)

    # 模拟收到 3 个 HTTP 请求
    requests = [
        {"rid": "req-1", "text": "Hello world", "max_tokens": 3},
        {"rid": "req-2", "text": "How are you today", "max_tokens": 2},
        {"rid": "req-3", "text": "Tell me a joke", "max_tokens": 4},
    ]

    for req in requests:
        print(f"[HTTP] 收到请求 rid={req['rid']}, text='{req['text']}'")
        sock.send_json(req)
        time.sleep(0.1)

    print("[HTTP] 所有请求已发送到 Tokenizer")
    sock.close()
    ctx.term()


def tokenizer_process(listen_addr: str, scheduler_addr: str):
    """模拟 TokenizerManager: 文本 → token IDs"""
    ctx = zmq.Context()

    # 接收来自 HTTP Server 的请求
    recv_sock = ctx.socket(zmq.PULL)
    recv_sock.bind(listen_addr)

    # 发送 tokenized 请求到 Scheduler
    send_sock = ctx.socket(zmq.PUSH)
    send_sock.connect(scheduler_addr)

    # 极简 tokenizer: 按空格切分，每个词映射为 hash
    def fake_tokenize(text):
        return [hash(word) % 10000 for word in text.split()]

    processed = 0
    while processed < 3:
        msg = recv_sock.recv_json()
        tokens = fake_tokenize(msg["text"])
        tokenized = {
            "rid": msg["rid"],
            "token_ids": tokens,
            "max_tokens": msg["max_tokens"],
        }
        print(f"[Tokenizer] rid={msg['rid']}: '{msg['text']}' → {tokens}")
        send_sock.send_json(tokenized)
        processed += 1

    recv_sock.close()
    send_sock.close()
    ctx.term()


def scheduler_process(listen_addr: str, detokenizer_addr: str):
    """模拟 Scheduler: 调度 + 假 forward"""
    ctx = zmq.Context()

    recv_sock = ctx.socket(zmq.PULL)
    recv_sock.bind(listen_addr)

    send_sock = ctx.socket(zmq.PUSH)
    send_sock.connect(detokenizer_addr)

    import random
    processed = 0
    while processed < 3:
        msg = recv_sock.recv_json()
        rid = msg["rid"]
        tokens = msg["token_ids"]
        max_tokens = msg["max_tokens"]

        print(f"[Scheduler] rid={rid}: prefill {len(tokens)} tokens")

        # 模拟 decode: 逐 token 生成
        output_tokens = []
        for i in range(max_tokens):
            new_token = random.randint(100, 999)
            output_tokens.append(new_token)
            time.sleep(0.02)  # 模拟 forward 延迟

        result = {"rid": rid, "output_tokens": output_tokens}
        print(f"[Scheduler] rid={rid}: generated {output_tokens}")
        send_sock.send_json(result)
        processed += 1

    recv_sock.close()
    send_sock.close()
    ctx.term()


def detokenizer_process(listen_addr: str):
    """模拟 DetokenizerManager: token IDs → 文本"""
    ctx = zmq.Context()

    recv_sock = ctx.socket(zmq.PULL)
    recv_sock.bind(listen_addr)

    # 极简 detokenizer
    vocab = {i: f"word_{i}" for i in range(1000)}

    processed = 0
    while processed < 3:
        msg = recv_sock.recv_json()
        rid = msg["rid"]
        text = " ".join(vocab.get(t % 1000, f"<{t}>") for t in msg["output_tokens"])
        print(f"[Detokenizer] rid={rid}: {msg['output_tokens']} → '{text}'")
        processed += 1

    recv_sock.close()
    ctx.term()


def main():
    print("=" * 60)
    print("ZMQ 多进程流水线 Demo")
    print("模拟: HTTP → Tokenizer → Scheduler → Detokenizer")
    print("=" * 60)
    print()

    # ZMQ 地址 (使用 TCP localhost)
    tok_addr = "tcp://127.0.0.1:15001"
    sch_addr = "tcp://127.0.0.1:15002"
    det_addr = "tcp://127.0.0.1:15003"

    # 启动 4 个进程 (模拟 SGLang 的多进程架构)
    procs = [
        multiprocessing.Process(target=detokenizer_process, args=(det_addr,), name="Detokenizer"),
        multiprocessing.Process(target=scheduler_process, args=(sch_addr, det_addr), name="Scheduler"),
        multiprocessing.Process(target=tokenizer_process, args=(tok_addr, sch_addr), name="Tokenizer"),
    ]

    # 按依赖顺序启动 (下游先启动)
    for p in procs:
        p.start()
        time.sleep(0.2)  # 等待 bind 完成

    # HTTP Server 最后启动 (发送请求)
    http_proc = multiprocessing.Process(target=http_server, args=(tok_addr,), name="HTTP")
    http_proc.start()

    # 等待所有进程完成
    http_proc.join()
    for p in procs:
        p.join(timeout=5)

    print()
    print("=" * 60)
    print("流水线完成!")
    print()
    print("对比 SGLang 源码:")
    print("  HTTP Server  → python/sglang/srt/entrypoints/http_server.py")
    print("  Tokenizer    → python/sglang/srt/managers/tokenizer_manager.py")
    print("  Scheduler    → python/sglang/srt/managers/scheduler.py")
    print("  Detokenizer  → python/sglang/srt/managers/detokenizer_manager.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

### 运行方式

```bash
python sglang-learning-docs/06_demo_zmq_pipeline.py
```

### 观察要点

1. **进程隔离**: 每个进程有独立的 ZMQ Context，通过地址通信
2. **启动顺序**: 下游 (bind) 先启动，上游 (connect) 后启动 — SGLang 也是这个顺序
3. **PUSH/PULL 模式**: 单向流水线，一对一传递
4. **对比源码**: SGLang 实际使用 `ipc://` 协议 (进程间共享内存)，延迟更低

---

## 实验 4: 自己动手改源码

在完成上面 3 个 demo 后，尝试以下修改：

### 4.1 给 Scheduler Demo 加上 Continuous Batching

修改实验 1 的代码，让 decode 批次中已完成的请求立即被移除，空出的位置立即被等待队列中的新请求填充。

提示：
```python
# 在 process_batch_result() 中，当一个请求完成时：
# 1. 从 running_batch 移除
# 2. 检查 waiting_queue 是否有新请求
# 3. 如果有，立即加入 running_batch (需要先 prefill)
```

### 4.2 给 RadixCache Demo 加上 LRU 淘汰

实现一个简单的 LRU 缓存淘汰：
- 维护一个 `last_access_time` 字典
- 当缓存满时，淘汰 `lock_ref=0` 且 `last_access_time` 最小的节点

### 4.3 给 ZMQ Demo 加上流式输出

修改实验 3，让 Scheduler 每生成一个 token 就发送给 Detokenizer（而不是攒完再发）。这就是 streaming 响应的原理。

---

## 总结: Demo 与源码的对应关系

| Demo | 模拟的组件 | 源码位置 | 简化了什么 |
|---|---|---|---|
| 实验 1 | Scheduler 主循环 | `scheduler.py:event_loop_normal()` | 去掉了 RadixCache、内存管理、overlap |
| 实验 2 | RadixCache | `radix_cache.py:RadixCache` | 简化了树节点结构和内存分配 |
| 实验 3 | 多进程流水线 | 4 个 Manager 进程 | 去掉了 overlap、批处理、错误处理 |
