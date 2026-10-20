# 插件与 registry：两种注册别混在一起

> 源码基准：上游 `16eb87c06f`；本地集成 `4f0e1cbea2`（2026-10-20）。本地日志增强与验证范围见总索引和 Mac 验证记录。

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

## 排查“装上了但没生效”

| 顺序 | 查询 |
|---|---|
| 1 | 当前 Python 环境是否装了正确发行包 |
| 2 | 发行包是否声明对应 entry_points 分组 |
| 3 | 选择与白名单是否允许加载 |
| 4 | hook/factory 是否实际执行，构造了什么对象 |

entry_points 类似由包元数据发现扩展；进程内 registry 则可能由 import 的副作用登记。不要把“注册名字存在”当成“配置已选择并执行”。

练习：解释为什么 kernel inventory 的多实现也不能自动证明最优实现已启用。答案：登记能力与选择/默认 wrapper 是不同契约。

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/plugins/__init__.py](../../python/sglang/srt/plugins/__init__.py) | entry_points 分组与过滤 |
| [python/sglang/srt/plugins/hook_registry.py](../../python/sglang/srt/plugins/hook_registry.py) | hooks |
| [python/sglang/srt/mem_cache/registry.py](../../python/sglang/srt/mem_cache/registry.py) | cache factory 注册 |
