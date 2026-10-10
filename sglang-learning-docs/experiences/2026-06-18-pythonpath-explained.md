# PYTHONPATH：确保运行当前检出的源码

> 源码基准：learning `2aa70e3eb4`，已合入 origin/main `6fc8d9da32`（2026-10-10）。

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

## 对照源码

| 文件 | 读什么 |
|---|---|
| [sglang-learning-docs/setup/launch_mac.sh](../setup/launch_mac.sh) | 设置源码路径 |
