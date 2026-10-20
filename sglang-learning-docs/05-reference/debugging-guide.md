# 排查：每个结论都落到一个证据

> 源码基准：上游 `16eb87c06f`；本地集成 `4f0e1cbea2`（2026-10-20）。本地日志增强与验证范围见总索引和 Mac 验证记录。

| 症状 | 第一条证据 | 下一步 |
|---|---|---|
| 无法连接 | 监听端口 / health | 确认进程是否启动完成 |
| MPS runtime 拒绝启动 | Torch 版本、MPS availability、堆栈 | 对照 runtime.py 门槛 |
| ModuleNotFoundError | 首个失败 import 和调用链 | 区分依赖缺失、平台分支和源码路径 |
| 输出不对 | 原始输入/输出 IDs、chat template、sampling | 区分模型、编码、采样和解码 |
| warm 没命中 | cached_tokens、输入 IDs、cache 类型 | 查 key、页面边界、有效组件、驱逐 |
| 请求卡住 | rid trace、queue、传输状态 | 查 admission 和生命周期 |
| 内存增加 | 活跃请求/可驱逐/受保护状态 | 区分缓存驻留与泄漏 |

## 常用命令

```bash
lsof -nP -iTCP:30000 -sTCP:LISTEN
curl --fail http://127.0.0.1:30000/health
curl --fail http://127.0.0.1:30000/server_info
uv pip check --python python/.venv/bin/python
PYTHONPATH=python python/.venv/bin/python - <<'PY'
import sglang, torch
print(sglang.__file__)
print(torch.__version__, torch.backends.mps.is_available())
PY
```

`sglang.__file__` 可以确认运行的是当前源码还是另一个安装包。多进程断点要 attach 到实际执行该函数的进程，不能只在启动器断点就期待进入 Scheduler。

## 区分参数输入与生效值

```text
启动命令 → 原始 ServerArgs → resolve / publish → 有效配置 bag
                                                ↓
                                          runner/cache 选择
```

检查 `get_exec/get_memory/get_schedule` 的读取位置、构造分支和日志。配置已发布后仅打印原始参数不足以证明覆盖生效。

## 本次真实失败

| 失败 | 已取证根因 | 处理 |
|---|---|---|
| 旧环境拒绝 MPS | Torch 2.9.1 低于 runtime 要求 | 按上游 srt_mps 升级 |
| proxy 连接拒绝 | 6152 无监听 | 直连下载成功 |
| pip check 冲突 | 旧 grpcio-tools 对 protobuf 约束不匹配 | 升级到兼容版本并重查 |
| 干净环境缺 mlx | Scheduler 的 MPS mixin 导入 mlx.core | 保留核心兼容包，固定标准 Torch 执行 |

细节和日志路径见 [验证记录](../setup/mac-validation.md)。端口历史服务已停止，后续复现需重新启动。

## 用排除链避免路径判断错误

```text
现象 → 当次原始输入/输出与日志 → 实际源码来源
    → 有效配置与实例类型 → 调用分支 → 状态/资源时序
    → 最小复现 → 最小修复 → 同场景回归
```

| 容易误判 | 应补证据 |
|---|---|
| cur_batch 为空，所以模型没执行 | 看赋值字段；本轮已修复旧 logger 字段不匹配 |
| 缓存越来越大，所以泄漏 | 区分正在用、可驱逐、Host 副本和延后释放 |
| 函数存在，所以新讲义正确 | 查返回类型、默认构造、调用顺序和实际分支 |
| 本地最近有提交，所以 SRT 是最新 | 单独比较 SRT 路径与固定上游；文档提交不改变推理代码 |
| setup 成功，所以已验证服务 | 实际启动并检查生成、协议和日志 |

本轮日志字段修复有旧实现失败、新实现通过的回归证据。历史 MPS 缺包与版本排查仍保留在日期化记录中；不能把历史失败当作新环境刚发生的事实。

## 对照源码

| 文件 | 读什么 |
|---|---|
| [python/sglang/srt/hardware_backend/mps/runtime.py](../../python/sglang/srt/hardware_backend/mps/runtime.py) | MPS 检查 |
| [python/sglang/srt/runtime_context.py](../../python/sglang/srt/runtime_context.py) | 生效配置 |
| [python/sglang/srt/managers/scheduler.py](../../python/sglang/srt/managers/scheduler.py) | 平台 import 与实际执行 |
| [sglang-learning-docs/setup/verify_mac.py](../setup/verify_mac.py) | 端点复现 |
