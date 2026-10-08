# 图像生成与尺寸记录

工具：内置 imagegen（scientific-educational）。主图以 my-sglang 为准，标准 SRT 差异写在图下。F1–F3 采用重新生成版本；F4–F6 采用原图经 imagegen 修正后的版本。

原生尺寸为 **1672×941**（F4 为 **1672×940**），交付尺寸均为 **2048×1152**。按用户确认，本地仅使用 Lanczos 等比例缩放、居中补白；未裁剪或修改图中文字和结构。放大不增加原生细节。

| 图 | 交付文件 | 原生尺寸 | 交付尺寸 |
|---|---|---|---|
| F1 | [f1-objects.png](f1-objects.png) | 1672×941 | 2048×1152 |
| F2 | [f2-request.png](f2-request.png) | 1672×941 | 2048×1152 |
| F3 | [f3-memory.png](f3-memory.png) | 1672×941 | 2048×1152 |
| F4 | [f4-forward.png](f4-forward.png) | 1672×940 | 2048×1152 |
| F5 | [f5-overlap.png](f5-overlap.png) | 1672×941 | 2048×1152 |
| F6 | [f6-ownership.png](f6-ownership.png) | 1672×941 | 2048×1152 |

## F1 · f1-objects.png

原生最终文件：`/Users/stream/.codex/generated_images/01a09f3d-1f0c-7f53-a563-c287d04230d7/exec-7c7d7454-126e-4517-a898-c7c12e1b25c1.png`

生成提示词：

```text
Use case: scientific-educational. Create a structured Chinese technical textbook infographic, landscape exactly 2048x1152 pixels. White background, crisp readable Chinese sans serif and exact English code labels. Blue=request and confirmed tokens; green=KV memory; purple=model; orange=asynchronous execution; red=recovery. Flat panels, generous whitespace, aligned rows, simple directional arrows, no decoration or people. Labels must match supplied text. This is my-sglang CPU teaching runtime, not real GPU performance. Keep technical text large. Title F1 核心对象关系. EXACT TEXT ONLY, no explanatory text invented. Draw 9 separate boxes in a clean directed graph. Blue boxes: 'Scheduler / waiting_queue / chunked_req / running_batch'; 'Req / rid / origin_input_ids / output_ids / status'; 'MiniScheduleBatch'; 'ForwardBatch'. Purple 'runner / forward / logits / sampling'. Green 'ReqToTokenPool / row,pos -> slot'; 'KV pool / K,V'; 'allocator / page'; 'radix cache / prefix / lock_ref'. Orange two separate smaller boxes 'FutureMap[row] / 下一轮输入 token'; 'result_queue / job,result,event'. Connect Scheduler -> MiniScheduleBatch label '选择 Req'; Req -> MiniScheduleBatch; MiniScheduleBatch -> ForwardBatch -> runner. runner -> Req label 'CPU 确认 output_ids'. allocator -> KV pool label '分配/回收'. Req -> ReqToTokenPool -> KV pool. radix cache -> ReqToTokenPool label '共享 slot'. runner -> FutureMap[row] -> runner labelled stash/gather. runner -> result_queue -> Req labelled 'launch 入队 / FIFO处理'. Footer 'F2 Req | F3 KV | F4 ForwardBatch | F5 overlap | F6 回收'. Do NOT nest MiniScheduleBatch within Req. Do NOT write rid anywhere in FutureMap. Do NOT label queue order completion order. Do NOT invent extra boxes or explanatory sentences.
```

## F2 · f2-request.png

原生最终文件：`/Users/stream/.codex/generated_images/01a09f3d-1f0c-7f53-a563-c287d04230d7/exec-47e2754e-7f82-44dc-adfd-ed086f1c58a6.png`

生成提示词：

```text
Use case: scientific-educational. Create a structured Chinese technical textbook infographic, landscape exactly 2048x1152 pixels. White background, crisp readable Chinese sans serif and exact English code labels. Blue=request and confirmed tokens; green=KV memory; purple=model; orange=asynchronous execution; red=recovery. Flat panels, generous whitespace, aligned rows, simple directional arrows, no decoration or people. Labels must match supplied text. This is my-sglang CPU teaching runtime, not real GPU performance. Keep technical text large. Title F2 Req 状态与生成历史. EXACT CONTENT ONLY. Four equal columns, labelled WAITING, RUNNING, RUNNING, FINISHED. Exactly three data rows: origin_input_ids with [7,8] in all columns; output_ids with [], [10], [10,11], [10,11,12]; get_fill_ids() with [7,8], [7,8,10], [7,8,10,11], [7,8,10,11,12]. Above three transition arrows labels 'EXTEND [7,8] -> 10', 'DECODE [10] -> 11', 'DECODE [11] -> 12'. Header 'F1 -> Req; 同步; max_new_tokens=3'. Bottom two panels exactly '中间 chunk: PREFILLING; output_ids 不变' and '停止 -> finished_reason'. Footer 'output_ids 保存确认历史；get_fill_ids() 是派生视图'. Do NOT draw ANY KV memory boxes in this image. No other text or objects.
```

## F3 · f3-memory.png

原生最终文件：`/Users/stream/.codex/generated_images/01a09f3d-1f0c-7f53-a563-c287d04230d7/exec-2f0849fa-e8d4-4414-b5de-4553e4a165df.png`

生成提示词：

```text
Use case: scientific-educational. Create a structured Chinese technical textbook infographic, landscape exactly 2048x1152 pixels. White background, crisp readable Chinese sans serif and exact English code labels. Blue=request and confirmed tokens; green=KV memory; purple=model; orange=asynchronous execution; red=recovery. Flat panels, generous whitespace, aligned rows, simple directional arrows, no decoration or people. Labels must match supplied text. This is my-sglang CPU teaching runtime, not real GPU performance. Keep technical text large. Title F3 token、位置与 KV 地址. EXACT CONTENT ONLY. Header 'F1 -> 内存; 同步 DECODE 后，未结束'. Left Req box 'req_pool_idx=0; output_ids=[10,11]'. Below one four-row table columns aligned with direct vertical arrows into cells: 'pos | 0 | 1 | 2'; 'token | 7 | 8 | 10'; 'slot | 2 | 3 | 4'; 'K/V | K/V(7) | K/V(8) | K/V(10)'. Under table two page blocks 'page1: slot2,slot3' and 'page2: slot4,slot5空'. RIGHT three row table '阶段 | allocated | committed' '准备前 | 2 | 2' '准备后 | 3 | 2' '成功后 | 3 | 3'. One red arrow preparation->before '失败回滚'. Footer '3个映射位置，占用4个slots；page_size=2'. Do NOT draw horizontal crossed mapping arrows. Do NOT write asynchronous. No additional labels.
```

## F4 · f4-forward.png

原生最终文件：`/Users/stream/.codex/generated_images/01a09f3d-1f0c-7f53-a563-c287d04230d7/exec-aec00421-562f-4417-aac5-dc70d2d31fe6.png`

生成提示词：

```text
Use case: scientific-educational. Create a structured Chinese technical textbook infographic, landscape exactly 2048x1152 pixels. White background, crisp readable Chinese sans serif and exact English code labels. Blue=request and confirmed tokens; green=KV memory; purple=model; orange=asynchronous execution; red=recovery. Flat panels, generous whitespace, aligned rows, simple directional arrows, no decoration or people. Labels must match supplied text. This is my-sglang CPU teaching runtime, not real GPU performance. Keep technical text large. F4 ForwardBatch 如何驱动模型. Breadcrumb 'F1 -> MiniScheduleBatch -> ForwardBatch -> runner'. Top two request rows: A row0 prefix2 seq3 new1 input[10] slot[4]; B row1 prefix0 seq2 new2 input[4,5] slots[6,7]. Main middle aligned strips exact 'input_ids = [10 | 4,5]', 'out_cache_loc = [4 | 6,7]', 'req_pool_indices = [0,1]', 'seq_lens = [3,2]', 'extend_seq_lens = [1,2]', 'extend_range_starts = [2,0]'. Bottom purple pipeline: Embedding -> Q/K/V + RoPE -> Attention -> FFN -> logits -> sampling. Green branch from Q/K/V says '新 K/V 写 out_cache_loc'; green branch into Attention 'row + seq_len 找历史 KV'. Arrow from sampling to blue box '每请求一个输出 -> CPU 更新 output_ids'. Footer '展平只包含本轮新输入；前缀从 KV 读取。示意省略 Norm、残差和输出投影。'. A is re-EXTEND or suffix input, both requests one EXTEND batch. Do not confuse B's token4 with A slot4.
```

最终修正提示词（以上一版图作为参考）：

```text
Change only two top-row labels 'out_cache_slot' to 'out_cache_loc'. Correct bottom explanatory write text to '按 out_cache_loc 写入新 K/V；extend_seq_lens 切分请求'. On Q/K/V+RoPE box add 'RoPE 只作用 Q、K'. Preserve all other exact data.
```

## F5 · f5-overlap.png

原生最终文件：`/Users/stream/.codex/generated_images/01a09f3d-1f0c-7f53-a563-c287d04230d7/exec-182ec8ff-a295-433a-a556-04365cbf199b.png`

生成提示词：

```text
Use case: scientific-educational. Create a structured Chinese technical textbook infographic, landscape exactly 2048x1152 pixels. White background, crisp readable Chinese sans serif and exact English code labels. Blue=request and confirmed tokens; green=KV memory; purple=model; orange=asynchronous execution; red=recovery. Flat panels, generous whitespace, aligned rows, simple directional arrows, no decoration or people. Labels must match supplied text. This is my-sglang CPU teaching runtime, not real GPU performance. Keep technical text large. F5 overlap 三处 token 记录. Breadcrumb 'F1 -> FutureMap / result_queue / Req'. Upper exact timeline with lanes CPU, forward stream, copy stream. CPU lane left to right 'enqueue B1' -> 'wait B0.copy_done' -> 'process B0'. forward lane 'B0 sample 10' -> 'stash FM[row]=10' -> 'B1 gather 10' -> 'B1 sample 11'; dependency ordered, NOT parallel forwards. copy lane 'B0 D2H' -> 'B0.copy_done' with arrow to CPU process. Bottom three distinct ledger panels: 'FutureMap[row]: 下一轮输入 token'; 'result_queue: job / result / event'; 'Req.output_ids: CPU 已确认历史'. Arrow stash to FutureMap; result queue to CPU process; CPU process to output_ids. Notes 'B0/B1 为稳定阶段 DECODE' and 'valid: stash=True，gather=False' and 'launch 成功推进 committed；不等于设备完成'. Footer '队列已有 job ≠ token 已算出 ≠ host 副本就绪'. Avoid labeling result_queue a token-history list.
```

最终修正提示词（以上一版图作为参考）：

```text
Fix inaccurate annotations in this diagram. Bottom result_queue panel: replace result column token=10/token=11 by '异步结果' and '待完成'; replace caption '由模型执行异步写入 (sample结果入队...)' with 'CPU launch 后按提交顺序入队'. Bottom output_ids caption must be 'process B0 后追加确认 token' not '推进 committed'. Remove the existing top timeline and replace with a simple dependency diagram: CPU row 'enqueue B1 -> wait B0.copy_done -> process B0'; forward row 'B0 sample10 -> stash10 -> B1 gather10 -> sample11'; copy row placed AFTER B0 sample/stash and connected from stash: 'B0 D2H -> B0.copy_done', arrow from copy_done to CPU process. Label '依赖示意，非耗时刻度；Fake CUDA 等 B0 时不执行 B1'. Do not draw D2H starting before sampling. Keep three ledger panels but avoid invented output position numbers by using simple output_ids '[...,10]' after process.
```

## F6 · f6-ownership.png

原生最终文件：`/Users/stream/.codex/generated_images/01a09f3d-1f0c-7f53-a563-c287d04230d7/exec-feed37f2-94c0-4d50-9aaf-932ae4d96525.png`

生成提示词：

```text
Use case: scientific-educational. Create a structured Chinese technical textbook infographic, landscape exactly 2048x1152 pixels. White background, crisp readable Chinese sans serif and exact English code labels. Blue=request and confirmed tokens; green=KV memory; purple=model; orange=asynchronous execution; red=recovery. Flat panels, generous whitespace, aligned rows, simple directional arrows, no decoration or people. Labels must match supplied text. This is my-sglang CPU teaching runtime, not real GPU performance. Keep technical text large. F6 缓存所有权与结束回收. Breadcrumb 'F1 -> radix / Req / allocator'. Top ownership diagram A row0 and B row1 both arrows into green shared page '[7,8] -> slots[2,3]' inside radix cache box 'lock_ref=2'. A additional private page '[9,10]' and B private tail '[11,空]'. Field labels 'prefix_indices: 借用的 slots', 'last_node: 锁定路径终点', 'cache_protected_len: 保护长度'. Bottom three columns: 'evict: 无锁 cache leaf -> 回收页'; 'retract: 释放 row / 私有KV -> WAITING; 保留 output_ids; retracted_stain=True; 重新 EXTEND'; 'finish: 停止生成 -> 完整已提交页交给 cache -> 释放私有资源'. Orange inset below finish 'overlap: 在途引用归零后再释放'. Footer 'cache_protected_len 记录长度；lock_ref 执行引用保护。启用 radix，page_size=2。'. Add small arrow from retract back to A labeled '逻辑历史保留，地址重新建立'. Do not imply evict frees locked page.
```

最终修正提示词（以上一版图作为参考）：

```text
Correct only these text errors, preserve structure: both A and B cache_protected_len must be 2, not 4, because shared prefix [7,8] has length2. Remove '(未提交)' from both private page boxes, private ownership does NOT mean uncommitted. Change 'row 状态 -> WAITING' to 'Req.status -> WAITING'. Change 'row 结束，清理运行时状态' to '释放 row，清理运行时状态'. In retract step1 text use '释放 row、私有 KV、runner 状态；解除借用锁'. Preserve correct labels elsewhere.
```
