# PYTHONPATH：确保运行当前检出的源码

> 源码基准：上游 `16eb87c06f`；本地集成 `4f0e1cbea2`（2026-10-20）。本地日志增强与验证范围见总索引和 Mac 验证记录。

```text
仓库根/python/sglang → PYTHONPATH 包含根/python → import sglang
```

```bash
PYTHONPATH=python python/.venv/bin/python - <<'PY'
import sglang
print(sglang.__file__)
PY
```

| 常见错误 | 结果 |
|---|---|
| 把 python/sglang 本身放进 PYTHONPATH | 包搜索层级不对 |
| 用系统 Python 代替 .venv Python | 依赖和版本不同 |
| 只确认 pip show | 不足以证明 import 的实际路径 |

launch_mac.sh 自动把绝对的仓库 python 路径放到 PYTHONPATH 前面。父进程传给子进程的环境帮助它们找到同一源码，但每个进程有独立的 Python 运行状态。

## 两个维度都要核对

```text
Python 可执行文件 → 用哪份依赖
sys.path / PYTHONPATH → 用哪份业务源码
```

虚拟环境正确，不代表源码一定来自工作区；源码路径正确，也不代表 Torch 版本正确。调试前同时打印 `sys.executable` 和 `sglang.__file__`。

练习：在其他目录执行 launch_mac.sh，为什么仍能找到仓库代码？答案：脚本从自身位置计算绝对根目录，再把根目录的 python 放在 PYTHONPATH 前部，而非依赖调用者当前目录。

## 对照源码

| 文件 | 读什么 |
|---|---|
| [sglang-learning-docs/setup/launch_mac.sh](../setup/launch_mac.sh) | 设置源码路径 |
