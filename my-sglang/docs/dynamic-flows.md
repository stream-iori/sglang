# 动态流程

`my-sglang` 有两条执行入口：`MiniScheduler.step()` 是同步基线；`MiniOverlapScheduler.pipeline_step()` 是 CPU Fake CUDA 的 SGLang overlap 主线。

```text
同步： schedule -> forward/sample -> CPU process

overlap：
schedule B1 -> forward_stream enqueue B1 -> copy_stream enqueue B1
                                          -> wait/process B0 -> FIFO pop B0
```

decode 输入始终按稳定 `req_pool_idx` 从 `MiniFutureMap.output_tokens_buf[row]` 获取。Fake runner 将 gather 留在 forward stream 的任务内；因此 B1 可在 CPU 尚未提交 B0 token 时先入队。

```text
FutureMap[row] = device token value
ReqToTokenPool[row] = KV slot mapping
```

二者只共享 row，不共享资源。请求结束后，最后一个 in-flight queue owner 退出，才依次 clear FutureMap row、释放 KV/row、调用 runner remove。

失败恢复丢弃全部尚未 FIFO 提交的 result，清 FutureMap 与物理资源；已经写入 `output_ids` 的前缀保留，未完成请求回到 waiting queue。
