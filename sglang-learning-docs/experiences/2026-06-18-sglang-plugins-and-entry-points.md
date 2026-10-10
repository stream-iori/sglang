# 插件与 registry：两种注册别混在一起

> 源码基准：learning `2aa70e3eb4`，已合入 origin/main `6fc8d9da32`（2026-10-10）。

| 机制 | 谁发现/选择 | 当前例子 |
|---|---|---|
| Python entry_points | 已安装发行包的 metadata | sglang.srt.platforms / sglang.srt.plugins |
| 运行时 registry | 进程内 factory / kernel metadata | cache backend / TreeCore / kernels |

```text
已安装插件 → load_plugins → 按选择/白名单加载 → hooks/platform
已注册 factory → 参数/配置选择 → 构造 cache 或解析 callable
```

不能说“pip 装上后一定所有插件无条件生效”：当前代码有 SGLANG_PLATFORM、SGLANG_PLUGINS 选择和过滤，也有加载失败处理。

排查插件先查发行包 entry_points、选择配置、加载日志，再查 hooks/实际实例。

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/plugins/__init__.py](../../python/sglang/srt/plugins/__init__.py) | entry_points 分组与过滤 |
| [python/sglang/srt/plugins/hook_registry.py](../../python/sglang/srt/plugins/hook_registry.py) | hooks |
| [python/sglang/srt/mem_cache/registry.py](../../python/sglang/srt/mem_cache/registry.py) | cache factory 注册 |
