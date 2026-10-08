# 全局地图生成记录

工具：内置 `image_gen`，按 imagegen 技能生成并检查。图用于建立空间关系；章节中的 ASCII、表格与源码引用用于核对精确语义。图中显存块是物理 KV 的概念示意，教学实现实际运行于 CPU。主图以普通完整 EXTEND 为例，06 的缓存保留以启用 radix cache 为前提。

目标尺寸：2048 × 1152。首次生成及一次明确要求尺寸修订后，工具均返回 1672 × 941；最终保存实际返回尺寸，未将其标作原生 2K。文件：`01-global-map.png`。图中 slot 数字是示意地址，与节点深入章节中的另一组分配示例不要求相同。

## 原始生成提示词

```text
Use case: scientific-educational
Asset type: my-sglang Chinese learning atlas, master image for linked Markdown chapters.
Primary request: Create a sharp 2K landscape 2048x1152 educational infographic, white background, generous whitespace, readable Chinese typography. Title "my-sglang：一次生成的全局地图". Diagram has exactly 6 numbered stages, arranged left to right in two connected rows of three:
top row "01 请求与调度" -> "02 KV 与地址" -> "03 ForwardBatch";
bottom row flows right to left: "04 模型计算" under 03 -> "05 确认输出" under 02 -> "06 结束与回收" under 01.
Use clear arrow from 03 down to 04. The spatial layout bottom labels from left to right 06,05,04. A labeled loop arrow from 05 back to 01 says "未结束：下一轮 DECODE".
Inside 01 show a request card "prompt [7,8]" and waiting queue trays. Inside 02 illustrate a logical-position address table connected to physical KV blocks, labels "row + position → slot" and "K / V". Inside 03 a neatly aligned pair of strips labelled "input_ids" and "out_cache_loc". Inside 04 show miniature computational chain "Embedding → Attention → FFN → logits" and "sampling → 10". Inside 05 a ledger "output_ids = [10]" and caption "下一轮输入 10". Inside 06 show returned empty memory blocks and a retained shared cache block, caption "释放私有资源".
A footer separated from the main flow has three small panels with labels "分块：长 prompt 分轮处理", "缓存：复用已有前缀 KV", "overlap：CPU 与执行流水交接". Small footer "教学运行时：CPU Fake CUDA / NumPy；不代表真实 GPU 性能".
Color identity consistent: requests and confirmed tokens blue; KV green; model purple; pipeline orange. Clean editorial textbook illustration, light isometric memory blocks and flat arrows; precise numbered stages; no characters or decorative logos. Do not suggest output token 10 is already in KV after prefill. No extra paths, no dense text.
```

## 尺寸修订提示词

```text
Use case: scientific-educational. Edit this existing my-sglang learning atlas. Preserve its six panels, exact Chinese text, arrows, colors, and diagram content. Change only image resolution: deliver a sharp 2K PNG at exactly 2048 x 1152 pixels, with readable text. Do not return the previous 1672 x 941 size. This is for zoomable technical documentation.
```
