# 找实际实现：从实例构造追到方法

> 源码基准：learning `2aa70e3eb4`，已合入 origin/main `6fc8d9da32`（2026-10-10）。

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

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/mem_cache/registry.py](../../python/sglang/srt/mem_cache/registry.py) | cache 工厂 |
| [python/sglang/srt/model_executor/model_runner.py](../../python/sglang/srt/model_executor/model_runner.py) | Runner 分支 |
