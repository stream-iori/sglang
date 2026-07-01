from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import TextIO

from my_sglang.models import BatchForward, Req, RequestStatus
from my_sglang.pools import KVPool, ReqPool, ReqToTokenMap
from my_sglang.runner import RunnerProtocol


@dataclass(frozen=True)
class StepResult:
    # step() 返回本轮实际做了什么，方便测试断言和人类观察。
    prefill_batch: BatchForward | None
    decode_batch: BatchForward | None
    finished_rids: tuple[str, ...]


class MiniScheduler:
    # MiniScheduler 是 normal scheduling 的核心。
    # 它维护两个队列：
    # 1. waiting_queue: 新请求，等待 prefill。
    # 2. running_reqs: 已经 prefill 完，等待一轮轮 decode。
    def __init__(
        self,
        runner: RunnerProtocol,
        *,
        max_running_reqs: int = 128,
        max_total_tokens: int = 8192,
        trace: bool = False,
        trace_file: TextIO | None = None,
    ):
        # runner 是真正执行模型 forward 的对象；可以是 fake runner，也可以是 MLX adapter。
        self.runner = runner
        # 等待被调度的Req,等待执行Prefill,没有KVCache
        self.waiting_queue: list[Req] = []
        # 处于prefill(Chunk) decode的Req
        self.running_reqs: list[Req] = []
        # 下面三个结构只模拟 SGLang 的资源管理关系，不存真实张量。
        self.req_pool = ReqPool(max_running_reqs)
        self.kv_pool = KVPool(max_total_tokens)
        self.req_to_token = ReqToTokenMap()
        self.trace = trace
        self.trace_file = trace_file or sys.stderr
        self.last_prefill_batch: BatchForward | None = None
        self.last_decode_batch: BatchForward | None = None

    def add_request(self, req: Req) -> None:
        # 新请求只能进入 waiting_queue，不能直接进入 running_reqs。
        # running 必须发生在 prefill 之后。
        if self.req_pool.get(req.rid) is not None or any(
            existing.rid == req.rid for existing in self.waiting_queue
        ):
            raise ValueError(f"duplicate rid {req.rid!r}")
        self.waiting_queue.append(req)
        self._emit("enqueue", rid=req.rid, prompt_len=len(req.origin_input_ids))

    def step(self) -> StepResult:
        # 记录本轮开始时已经在 running 的请求。
        # 这样本轮新 prefill 的请求不会立刻 decode，避免同一步里跑两次模型。
        running_at_step_start = list(self.running_reqs)
        finished: list[str] = []
        # normal scheduling 的一轮：先接纳 waiting 做 prefill，再 decode 老的 running。
        prefill_batch = self._run_prefill_waiting(finished)
        decode_batch = self._run_decode(running_at_step_start, finished)

        # 保留最近一次非空 batch，便于集成测试和调试观察。
        if prefill_batch is not None:
            self.last_prefill_batch = prefill_batch
        if decode_batch is not None:
            self.last_decode_batch = decode_batch
        return StepResult(prefill_batch, decode_batch, tuple(finished))

    def run_until_complete(self) -> list[Req]:
        # 简单阻塞式运行：只要还有 waiting 或 running 请求，就不断 step。
        completed: list[Req] = []
        while self.waiting_queue or self.running_reqs:
            # before 用来在资源释放后仍能找到刚完成的 Req 对象。
            # [*,*]两个list合并成一个新的list
            # 左边req.rid是key
            # {req.rid, req(item) | for req}
            before = {req.rid: req for req in [*self.waiting_queue, *self.running_reqs]}
            result = self.step()
            for rid in result.finished_rids:
                if rid in before:
                    completed.append(before[rid])
        return completed

    def _run_prefill_waiting(self, finished: list[str]) -> BatchForward | None:
        if not self.waiting_queue:
            return None

        # 取出当前所有 waiting 请求，拼成一个 prefill batch。
        reqs = self.waiting_queue
        self.waiting_queue = []
        # planned 保存已经申请成功的资源；如果中途失败，需要靠它回滚。
        planned: list[tuple[Req, int, list[int]]] = []

        try:
            for req in reqs:
                # 一个请求先占用一个 req_pool_idx，再为 prompt 的每个 token 申请 KV slot。
                req_pool_idx = self.req_pool.alloc(req.rid)
                try:
                    slots = self.kv_pool.alloc_many(len(req.origin_input_ids))
                except Exception:
                    # 如果 KV slot 申请失败，要立刻归还刚申请的 req slot。
                    self.req_pool.free(req.rid)
                    raise
                planned.append((req, req_pool_idx, slots))
        except Exception:
            # 资源申请阶段要求“全有或全无”：任意失败都回到 step 前状态。
            for req, req_pool_idx, slots in reversed(planned):
                self.kv_pool.free_many(slots)
                self.req_to_token.remove_req(req_pool_idx)
                self.req_pool.free(req.rid)
                req.req_pool_idx = None
                req.kv_slots.clear()
            self.waiting_queue = reqs + self.waiting_queue
            raise

        batch = BatchForward(
            # BatchForward 只是把调度结果结构化，方便理解和测试。
            mode="prefill",
            reqs=tuple(req for req, _, _ in planned),
            input_ids_by_req=tuple(
                tuple(req.origin_input_ids) for req, _, _ in planned
            ),
            req_pool_indices=tuple(req_pool_idx for _, req_pool_idx, _ in planned),
            out_cache_locs=tuple(tuple(slots) for _, _, slots in planned),
            seq_lens=tuple(len(req.origin_input_ids) for req, _, _ in planned),
        )
        self._emit(
            "prefill_batch",
            rids=[req.rid for req in batch.reqs],
            seq_lens=list(batch.seq_lens),
        )

        for req, req_pool_idx, slots in planned:
            # 把资源归属写回 Req，后续 finish 时才能释放。
            req.req_pool_idx = req_pool_idx
            req.kv_slots.extend(slots)
            # 记录 prompt 每个位置对应哪个 KV slot。
            for offset, slot in enumerate(slots):
                self.req_to_token.set(req_pool_idx, offset, slot)

            # prefill 输入整个 prompt，返回第一个生成 token。
            token = self.runner.prefill(
                req_id=req.rid,
                new_token_ids=list(req.origin_input_ids),
                full_token_ids=list(req.origin_input_ids),
                prefix_slot_ids=[],
                new_slot_ids=list(slots),
                req_pool_idx=req_pool_idx,
            )
            req.append_output(token)
            if req.maybe_finish():
                # 如果 prefill 直接生成 eos，或者 max_new_tokens=1，就不进入 running。
                self._finish_req(req, finished)
            else:
                req.mark_running()
                self.running_reqs.append(req)
                self._emit("prefill_done", rid=req.rid, next_token=token)

        return batch

    def _run_decode(
        self,
        candidates: list[Req],
        finished: list[str],
    ) -> BatchForward | None:
        decode_reqs = [
            req
            for req in candidates
            if req.status is RequestStatus.RUNNING and req in self.running_reqs
        ]
        if not decode_reqs:
            return None

        # decode 每个请求只需要为“本轮输入 token”申请一个新的 KV slot。
        allocated: list[tuple[Req, int]] = []
        try:
            slots = self.kv_pool.alloc_many(len(decode_reqs))
            allocated = list(zip(decode_reqs, slots, strict=True))
        except Exception:
            self.kv_pool.free_many([slot for _, slot in allocated])
            raise

        batch = BatchForward(
            mode="decode",
            reqs=tuple(decode_reqs),
            # decode 阶段输入的是每个请求当前最后一个 token。
            input_ids_by_req=tuple((req.last_token_id,) for req in decode_reqs),
            req_pool_indices=tuple(
                self._require_req_pool_idx(req) for req in decode_reqs
            ),
            out_cache_locs=tuple((slot,) for _, slot in allocated),
            seq_lens=tuple(len(req.full_token_ids) for req in decode_reqs),
        )
        self._emit(
            "decode_batch",
            rids=[req.rid for req in batch.reqs],
            input_ids=[ids[0] for ids in batch.input_ids_by_req],
        )

        for req, slot in allocated:
            req_pool_idx = self._require_req_pool_idx(req)
            req.kv_slots.append(slot)
            # 这里记录的是 decode 输入 token 的 KV 位置。
            # append_output 发生在后面，所以当前 len(full_token_ids)-1 正好是输入 token 的序列位置。
            self.req_to_token.set(req_pool_idx, len(req.full_token_ids) - 1, slot)

        # 真正批量调用模型：多个 request id 一起进入 runner.decode_batch。
        next_tokens = self.runner.decode_batch([req.rid for req in decode_reqs])
        if len(next_tokens) != len(decode_reqs):
            raise RuntimeError(
                f"runner returned {len(next_tokens)} tokens for {len(decode_reqs)} reqs"
            )

        for req, token in zip(decode_reqs, next_tokens, strict=True):
            # runner 返回的 token 是下一轮 decode 的输入，也会被追加到 output_ids。
            req.append_output(token)
            if req.maybe_finish():
                self._finish_req(req, finished)
            else:
                self._emit("decode_done", rid=req.rid, next_token=token)

        return batch

    def _finish_req(self, req: Req, finished: list[str]) -> None:
        # 统一释放请求相关资源：running 队列、req_to_token 映射、ReqPool、KVPool、runner 内部状态。
        req.status = RequestStatus.FINISHED
        if req in self.running_reqs:
            self.running_reqs.remove(req)
        if req.req_pool_idx is not None:
            self.req_to_token.remove_req(req.req_pool_idx)
        self.req_pool.free(req.rid)
        self.kv_pool.free_many(req.kv_slots)
        req.kv_slots.clear()
        self.runner.remove_request(req.rid)
        finished.append(req.rid)
        self._emit("finish", rid=req.rid, reason=req.finish_reason)

    def _require_req_pool_idx(self, req: Req) -> int:
        # 类型保护：对已经进入模型执行阶段的请求，req_pool_idx 必须存在。
        if req.req_pool_idx is None:
            raise RuntimeError(f"request {req.rid} has no req_pool_idx")
        return req.req_pool_idx

    def _emit(self, event: str, **payload) -> None:
        # trace 使用 JSON lines，方便人读，也方便后续脚本分析调度过程。
        if not self.trace:
            return
        print(
            json.dumps({"event": event, **payload}, sort_keys=True),
            file=self.trace_file,
        )
