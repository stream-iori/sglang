# Triton 基础概念

本文解释阅读 `my-sglang/examples/triton` 时容易混淆的 Triton 概念。program、tile、
logical lane 与 CUDA 执行模型的对应关系见
[Triton 与 CUDA：代码概念映射](triton-cuda-basics.md)。
如果还不熟悉 Tensor、shape、dimension、stride、view 和 broadcasting，先看
[PyTorch 基础概念](pytorch-concept.md)；本文只继续解释它们如何影响 Triton pointer 寻址。

## contiguous 为什么能让多维 tensor 按一维 offset 访问

[`02_fused_elementwise.py`](../examples/triton/02_fused_elementwise.py) 中有下面的注释：

```python
# contiguous 保证把任意输入 shape 展平成一维后，offset 仍对应连续元素。
```

它的意思是：这个 Triton kernel 不理解 tensor 的行、列等维度，只把 tensor 当成一段连续
内存，用下面的方式访问：

```python
offsets = 0, 1, 2, ..., x.numel() - 1
x = tl.load(x_ptr + offsets)
```

实际代码会把这些 offsets 分给多个 program，但所有 program 合起来仍然覆盖
`0..x.numel()-1`。kernel 只接收 `x_ptr` 和 `n_elements`，没有接收多维 shape 以及每一维的
stride，所以设备代码无法根据行、列坐标计算地址。

假设 `x.shape == [2, 3]`：

```text
逻辑形状：
[[a, b, c],
 [d, e, f]]

连续内存：
a, b, c, d, e, f
```

此时用 `x_ptr + 0..5` 依次读取，等价于将 tensor 按逻辑顺序展平为：

```text
[a, b, c, d, e, f]
```

这里的“展平”只是 kernel 的访问视角。代码没有调用 `flatten()`，也没有创建或复制一个新的
一维 tensor。

### 非连续 tensor 为什么可能出错

转置等操作可以只修改 tensor 的 shape 和 stride，而不重新排列底层存储。例如：

```python
x = original.T
```

转置后 tensor 的逻辑展平顺序可能是：

```text
[a, d, b, e, c, f]
```

底层存储顺序却仍然是：

```text
[a, b, c, d, e, f]
```

如果 kernel 继续使用 `x_ptr + 0..5`，它只能按照底层存储顺序读取，不能得到 tensor 的逻辑
展平顺序。要正确支持这种非连续布局，kernel 必须接收各维 stride，并根据逻辑坐标计算地址；
`03_row_softmax.py` 等二维示例中的 `row_idx * input_row_stride` 就是在显式完成这类寻址。

示例 02 选择更简单的做法：launcher 要求 `x` 和 `bias` 都满足：

```python
x.is_contiguous()
bias.is_contiguous()
```

`is_contiguous()` 只检查布局，不会自动转换或复制。检查失败时，本示例直接抛出异常。如果业务
代码希望接受非连续输入，可以先调用 `x.contiguous()` 得到逻辑内容相同、底层存储连续的新
tensor，但这可能产生一次额外复制和内存开销。

因此，原注释更完整地写就是：

```python
# kernel 不读取多维 shape/stride，而是用 ptr + 0..numel-1 线性访问 tensor。
# contiguous 保证这种内存访问顺序与 tensor 的逻辑展平顺序一致。
```

需要区分两个概念：

| 概念 | 含义 |
|---|---|
| 逻辑展平顺序 | 按 tensor 当前 shape 和 stride 观察到的元素顺序。 |
| 底层存储顺序 | 从数据指针开始按照连续地址实际排列的元素顺序。 |

连续 tensor 的这两个顺序一致，所以一维 `ptr + offset` 足以正确访问任意逻辑 shape；非连续
tensor 的两个顺序可能不同，kernel 就不能在不知道 stride 的情况下把它当作连续数组处理。
