# 贯穿实验（二）：三个请求共享有限 KV

[返回全局地图](README.md) · 上一页：[状态账本](12-request-ledger.md) · 下一页：[验证实验](14-evidence-lab.md)

**先让 A、B 共享并触发 retract，再让 C 触发无锁缓存淘汰。** 这是一条连续可运行的时间线。若提前让 C 在资源紧张时入队，当前教学准入可能 abort C，因此不将“三请求同时运行”当作本实验前提。

```bash
uv run python examples/runtime_walkthrough.py shared
```

## 固定参数与输入

| 参数 | 值 |
|---|---|
| 可用容量 | 8 slots，page_size=2，即 4 页；padding page 0 不计入 |
| chunk size / request rows | 2 / 3 |
| radix / new_token_ratio | 开启 / 0，刻意放松初始预留以观察压力 |
| A | prompt `[7,8,9,10]`，输出上限 3，预设输出 40、41、42 |
| B | prompt `[7,8,11]`，输出上限 3，预设输出 50、51、52 |
| C | prompt `[20,21,22,23,24,25]`，输出上限 1，预设输出 60 |

runner 按请求身份和已确认输出数产生编号；中间 chunk 返回值被调度器忽略。它只验证状态和地址，不验证模型数值或采样质量。

## 图 S17：实跑十轮

| step | 本轮输入 | 发生什么 |
|---:|---|---|
| 1 | A EXTEND `[7,8]` | 中间 chunk，完整页进入 cache，A 尚无输出 |
| 2 | A EXTEND `[9,10]` | B 已入队但等待；A 首输出 40 |
| 3 | B EXTEND `[11]` | 借用 `[7,8]`，只算 suffix；B 首输出 50 |
| 4 | A DECODE 40，B DECODE 50 | 输出 41、51；池已满 |
| 5 | B DECODE 51 | 先 retract A，B 输出 52 并结束 |
| 6 | A EXTEND `[9,10]` | A 重新准入，继续借用 `[7,8]`，保留输出 `[40,41]` |
| 7 | A EXTEND `[40,41]` | 恢复完整上下文，产生 42 并结束 |
| 8 | C EXTEND `[20,21]` | C 此时才入队，淘汰旧无锁叶页并复用 |
| 9 | C EXTEND `[22,23]` | 继续淘汰旧叶页 |
| 10 | C EXTEND `[24,25]` | 首输出 60，达到上限并结束 |

## 图 S18：逐页看所有权变化

P1=`[2,3]`，P2=`[4,5]`，P3=`[6,7]`，P4=`[8,9]`。方括号里的 token 仅标记其 KV 所属位置。

| step 后 | P1 | P2 | P3 | P4 |
|---:|---|---|---|---|
| 1 | cache `[7,8]`，A 锁 | 空闲 | 空闲 | 空闲 |
| 2 | cache，A 锁 | A 私有 `[9,10]` | 空闲 | 空闲 |
| 3 | cache，A/B 共锁 | A 私有 | B 私有 `[11,空]` | 空闲 |
| 4 | cache，A/B 共锁 | A 私有 | B 私有 `[11,50]` | A 私有 `[40,空]` |
| 5 | cache 无锁 | 空闲 | cache `[11,50]` 无锁 | 空闲 |
| 6 | cache，A 锁 | 空闲 | cache 无锁 | cache `[9,10]`，A 锁 |
| 7 | cache 无锁 | cache `[40,41]` 无锁 | cache 无锁 | cache 无锁 |
| 8 | cache 无锁 | cache 无锁 | C/cache `[20,21]` 锁 | cache 无锁 |
| 9 | cache 无锁 | C/cache `[22,23]` 锁 | C/cache 锁 | cache 无锁 |
| 10 | 旧 cache `[7,8]` 无锁 | C/cache 无锁 | C/cache 无锁 | C/cache `[24,25]` 无锁 |

step5 的 P2 曾临时用于 B 输入 51，B 当轮结束后，不完整尾页随即释放，因此表中是空闲。表格统一观察 step 返回后，避免混合临时状态。

最后 `[7,8]` **仍然保留**；C 只需 3 页，淘汰旧叶页并不必清空整棵旧树。末次查询旧 `[7,8,9,10]` 仍命中 2 tokens，C 的 prompt 命中 6 tokens，脚本均有断言。

## 地址数与实际占用不是同一统计

| 时刻 | mapped_tokens | allocated_tokens | 原因 |
|---|---:|---:|---|
| step3 | 7 | 6 | A 映射 4 个、B 映射 3 个，共用两个 slots；另有尾页空位 |
| step4 | 9 | 8 | 两条请求地址表重复引用共享前缀 |
| step7 | 0 | 8 | 请求已结束，8 slots 由无锁 cache 持有 |

## 图 S19：retract 恢复的到底是什么

```text
A 原 row=0，输出 [40,41]
     |
retract：释放私有资源，row=None，输出不变
     |
再次准入：row=2，匹配 [7,8]
     |
EXTEND [9,10] -> EXTEND [40,41] -> 新输出 42
```

重新计算旧 token 的 KV，不等于重新向用户生成或确认旧 token。row 可换，已确认的逻辑历史不变。

代码：[共享案例](../../examples/runtime_walkthrough.py)、[缓存交接与 retract](../../src/my_sglang/scheduler.py)、[radix 页树](../../src/my_sglang/radix_cache.py)。
