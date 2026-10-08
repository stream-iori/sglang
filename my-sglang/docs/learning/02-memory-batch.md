# 02～03 KV 与 ForwardBatch：先有地址，再执行

配套大图：[F3 地址与水位、F4 模型工作单](15-field-atlas.md)。

[返回全局地图](README.md) · 上一页：[请求](01-request.md) · 下一页：[模型](03-model.md)

**token id、逻辑位置、物理 slot 是三种不同数字。** Scheduler 分配地址，模型按地址写 K/V。

## 02：用一张表接上逻辑和物理

```text
请求 A，row=3 ( req_pool_idx = 3)
逻辑位置 pos             0       1       2
输入 token               7       8      10
req_to_token[3,pos]      8       9      12
                         |       |       |
物理存储               KV[8]   KV[9]   KV[12]
                      token7   token8  token10 的 K/V
```

| 对象 | 保存什么 | 由谁使用 |
|---|---|---|
| Req.output_ids | `[10,11,...]` 这样的已确认结果 | CPU 更新请求状态 |
| ReqToTokenPool | `[row,pos] → slot` 的二维地址表 | 模型找历史 K/V，调度器管理映射 |
| allocator | slot/page 的分配与回收状态 | Scheduler 申请或释放空间 |
| 物理 KV pool | slot 对应的 K/V 向量 | Attention 读写 |

一个 slot 容纳一个 token 所需的 K/V 位置；page 将固定数量的 slot 作为分配单位。row 空闲、page 空闲、私有尾页尚有位置，三者不同。

```text
page_size=2
已处理 [7,8]： [slot8:7 | slot9:8]    -> 尾页已满
输入 10：     [slot8:7 | slot9:8] [slot12:10 | slot13:空]
输入 11：     [slot8:7 | slot9:8] [slot12:10 | slot13:11]
                                      ^复用自己的尾页，无需新 page
```

图中 `slot:token` 只是说明 K/V 属于哪个 token。token 10 需要新 page 是因为旧尾页满了；若分配器没有空闲 page，才会触发内存压力。

## 两个 KV 水位：拿到地址，不等于完成计算

| 同步处理 prompt `[7,8]` | allocated | committed | output_ids |
|---|---:|---:|---|
| 准备前 | 0 | 0 | `[]` |
| 已分配 slot，尚未 forward | 2 | 0 | `[]` |
| runner 成功，提交并处理输出 | 2 | 2 | `[10]` |
| 如果 runner 失败并回滚 | 0 | 0 | `[]` |

`allocated` 是已分配位置的边界；`committed` 是调度器已提交的边界。**同步在 runner 成功返回后提交；overlap 在异步 launch 成功后提交，不能由 committed 推断设备已经完成。** 这两个字段都不是“是否搬进 HBM”的标志。

回滚顺序的目的：清除未提交映射，释放不与 committed 部分共页的新 page，再把 allocated 拉回 committed。若失败位置和稳定历史共用尾页，不能把整个尾页释放。

## 03：ForwardBatch 是本轮工作单

A 的输入 `[7,8,10]` 已有前两个 token 的 KV；B 的新 prompt 是 `[4,5]`。假定本轮分配 slot 如下：

| 请求 | prefix_len | seq_len | extend_len | 本轮输入 | 新 KV slot |
|---|---:|---:|---:|---|---|
| A | 2 | 3 | 1 | `[10]` | `[12]` |
| B | 0 | 2 | 2 | `[4,5]` | `[20,21]` |

```text
input_ids          = [10 | 4,5]    <- 本轮输入
out_cache_loc      = [12 | 20,21]  <- 与输入逐项对齐(新KV Slot)
extend_seq_lens    = [1,2]         <- 用长度切回 A、B(这是一个技术变量)
seq_lens           = [3,2]         <- 每请求本轮结束后的总长度
extend_range_starts= [2,0]         <- 本轮在各请求里的起点
```

展平把变长请求的新 token 接成一个输入序列，历史前缀通过地址表访问，不重复塞进 input_ids。`MiniScheduleBatch` 的字段叫 `extend_lens`，交给 runner 的 `ForwardBatch` 字段叫 **`extend_seq_lens`**。

### 例子：一条含 7 个 token 的 prompt，分两轮处理

下面是一个独立于上面 A、B 的例子。用户提交一条请求，prompt 分词后得到 `[21,22,23,24,25,26,27]`，共 7 个 token。这些数字是输入的 token id，不是模型生成的答案。

假设调度器采用 chunked prefill（把长输入分块计算），第一轮只安排处理前 4 个 token，第二轮再处理剩下的 3 个。这里的 4 是为了举例选的分块长度，不是模型的固定规则。

| 轮次 | 送进模型的输入 token | 这一轮做完后 |
|---|---|---|
| 第一轮 | `[21,22,23,24]` | 这 4 个 token 的各层 K/V 已算出并保存；prompt 尚未处理完，不确认答案 token |
| 第二轮（下面说的“本轮”） | `[25,26,27]` | 在本例的全历史因果 Attention 中，3 个新位置各自都读取前 4 个位置的 K/V，并计算、保存自己的新 K/V；完整 prompt 处理完，可以采样第一个答案 token |

**现在站在第二轮开始前看：已有 KV 的 token 有 4 个，还需送进模型计算的有 3 个，两部分合起来是整条 prompt 的 7 个 token。** 这才是下面 `4 + 3 = 7` 的来由。

```text
origin_input_ids = [21,22,23,24,25,26,27]
output_ids       = []
get_fill_ids()   = origin_input_ids + output_ids
                 = [21,22,23,24,25,26,27]

逻辑位置（从 0 开始）    0    1    2    3  |  4    5    6
token id              21   22   23   24  | 25   26   27
本轮开始时的 KV       已有 已有 已有 已有 | 待算 待算 待算
                     <--- prefix=4 ---> | <- extend=3 ->
                     <--------- seq_len=7 ----------->
```

| 名字 | 这个例子里的含义 | 数值 |
|---|---|---:|
| `prefix_len` | 本轮起点之前已有可用 KV 的位置数 | 4 |
| `extend_len` | 本轮新增计算的位置数 | 3 |
| `seq_len` | 本轮处理范围的结束边界，也就是处理后覆盖的总长度 | 7 |

因此 `prefix_len + extend_len = seq_len`，就是 **已有 4 个位置 + 新算 3 个位置 = 覆盖前 7 个位置**。这些数是长度或边界，不是 token id，也不是物理 slot。

### `[4:7]` 到底取出了什么？

Python 切片包含左边界、不包含右边界。所以 `[4:7]` 取的是位置 **4、5、6**，即第 **5、6、7** 个 token。

```python
fill_ids = [21, 22, 23, 24, 25, 26, 27]
fill_ids[4:7]  # [25, 26, 27]
```

对应到本轮只有这一个请求的 ForwardBatch：

| 字段 | 值 | 用途 |
|---|---|---|
| `input_ids` | `[25,26,27]` | 本轮实际送进模型的新输入 |
| `extend_range_starts` | `[4]` | 新输入从请求的逻辑位置 4 开始 |
| `extend_seq_lens` | `[3]` | 本轮有 3 个新输入 |
| `seq_lens` | `[7]` | Attention 查 KV 时使用的总范围边界 |
| `out_cache_loc` | 3 个物理 slot | 按顺序存放 token 25、26、27 的新 K/V |

### 前 4 个没放进 `input_ids`，模型怎么知道它们？

**通过 KV cache。** 每层只为新位置计算 Q/K/V，并写入新 K/V；Attention 同时读取已有前缀和本轮允许看到的新位置的 K/V。

```text
前 4 个 token：21、22、23、24
        |
        +-- 前一轮已算好各层 K/V，按地址表保存在 KV pool
                                      |
本轮 input_ids=[25,26,27]             |
        |                             |
        +-- 计算新的 Q/K/V            |
        +-- 写入新的 K/V              |
        +-- Attention <---------------+ 读取前缀 K/V
```

因果 Attention 仍然不能偷看后面的 token：

这里限定为 my-sglang **tiny 模型的全历史因果 Attention**：位置 `p` 读取位置 `0` 到 `p`（包含自己）的 K/V，所以本例三个新位置一定都会读取前 4 个位置的 K/V。源码先通过 `req_to_token_pool.row(row, position + 1)` 取这些位置的 slot，再读取 `key_cache[history]` 和 `value_cache[history]`，见 [`tiny_transformer.py`](../../src/my_sglang/tiny_transformer.py)。这不是由 EXTEND 这个调度名称决定的，而是由模型的 Attention 可见范围决定的；换成只允许看局部窗口的模型，就不能直接套用这个结论。

| 本轮计算的位置 | 可以读取哪些位置的 K/V |
|---|---|
| pos4：token 25 | pos0～4：前 4 个 + 自己 |
| pos5：token 26 | pos0～5：前 4 个 + 25、26 |
| pos6：token 27 | pos0～6：前 4 个 + 25、26、27 |

所以“不再次处理全部 7 个”指**不重新为前 4 个位置执行整套模型计算**；它们的 K/V 仍会参与新位置的 Attention。本轮有 3 个新输入，也不代表生成 3 个输出：这个例子补完 prompt 后，通常从最后位置的 logits 采样第一个输出 token。

### 两个容易混淆的边界

| 问题 | 怎么理解 |
|---|---|
| `prefix_len` 就是原始 prompt 长度吗？ | 不是。这里 prompt 长 7，已有 KV 的前缀长 4；前缀可来自之前的 chunk，也可来自缓存复用。 |
| `seq_len` 总等于 `len(get_fill_ids())` 吗？ | 不一定。长 prompt 分块时，假设共有 10 个 token，本轮只做到位置 6，仍然可以是 `4+3=7`，位置 7～9 留给后续轮次。 |

源码中的对应关系是：`prefix_len = extend_range.start`、`seq_len = extend_range.end`、`extend_len = end - start`，实际输入取 `req.get_fill_ids()[start:end]`，见 [`prepare_for_extend()`](../../src/my_sglang/schedule_batch.py)。

`MiniScheduleBatch` 是可变的准备单；`ForwardBatch` 把本轮数值元数据固定下来，但其中的 Req 引用仍指向可变请求，不能理解成整棵对象深拷贝。

自检：A 的新 token 为什么写 slot 12，而不是 token id 10 对应的地址？因为地址由 allocator 和映射决定，token id 不是内存地址。

深入：[数据结构](../data-structures.md)、[连接层](../model-execution-bridge.md)。源码：[`prepare / commit / rollback`](../../src/my_sglang/schedule_batch.py)、[`ForwardBatch`](../../src/my_sglang/models.py)。
