# 找实际实现：从实例构造追到方法

> 源码基准：上游 `16eb87c06f`；本地集成 `4f0e1cbea2`（2026-10-20）。本地日志增强与验证范围见总索引和 Mac 验证记录。

```text
调用 base.method
   → 实例是什么类？
   → 工厂在哪个条件下构造它？
   → 该类或继承链在哪实现 method？
```

```bash
rg -n 'class .*BasePrefixCache|def match_prefix' python/sglang/srt/mem_cache
rg -n 'default_radix_cache_factory|create_unified_radix_cache' python/sglang/srt/mem_cache/registry.py
```

只搜方法名会找到多个实现。当前普通路径的缓存需要先查 registry；执行器要先看 ModelRunner 路由，不能把搜到的第一个 forward 当实际调用。

## 从 Java 接口习惯迁移

Java 也需要知道接口背后的实际实例。Python 更常通过 factory、registry 和局部 import 选择类，所以 IDE 的“跳到实现”只是候选列表。

练习：从 `tree_cache.match_prefix` 往回找本地构造。答案要包含 registry 条件、UnifiedRadixCache、组件/TreeCore 选择，再用启动日志确认；只写一个继承类名不合格。

再选 ModelRunner 的执行器，用同样方法查 eager 与 Graph，比较“有这个类”与“此次调用实例”的区别。

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/mem_cache/registry.py](../../python/sglang/srt/mem_cache/registry.py) | cache 工厂 |
| [python/sglang/srt/model_executor/model_runner.py](../../python/sglang/srt/model_executor/model_runner.py) | Runner 分支 |
