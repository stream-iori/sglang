from __future__ import annotations

from collections import deque


class ReqPool:
    # ReqPool 模拟 SGLang 里的请求槽位池。
    # 每个活跃请求占用一个整数 idx，用它作为 req_to_token 映射的第一维。
    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("ReqPool capacity must be positive")
        # deque 适合做 FIFO 队列；这里存放当前可复用的请求槽位编号。
        self._free = deque(range(capacity))
        # rid -> idx，便于通过请求 ID 查到它占用的槽位。
        self._rid_to_idx: dict[str, int] = {}

    def alloc(self, rid: str) -> int:
        # 同一个 rid 同时只能有一个活跃请求，避免资源归属混乱。
        if rid in self._rid_to_idx:
            raise ValueError(f"duplicate rid {rid!r}")
        if not self._free:
            raise RuntimeError("ReqPool exhausted")
        idx = self._free.popleft()
        self._rid_to_idx[rid] = idx
        return idx

    def free(self, rid: str) -> None:
        # pop(..., None) 让重复释放成为无害操作，简化异常清理路径。
        idx = self._rid_to_idx.pop(rid, None)
        if idx is not None:
            self._free.append(idx)

    def get(self, rid: str) -> int | None:
        return self._rid_to_idx.get(rid)

    @property
    def active_count(self) -> int:
        return len(self._rid_to_idx)


class KVPool:
    # KVPool 模拟 token_to_kv_pool：每个 token 会占用一个 KV cache slot。
    # 这里不存真正的 KV 张量，只追踪 slot 生命周期，帮助理解调度逻辑。
    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("KVPool capacity must be positive")
        self._free = deque(range(capacity))
        self._allocated: set[int] = set()

    def alloc_many(self, count: int) -> list[int]:
        # prefill 会一次申请 prompt 长度个 slot；decode 每个请求每轮申请 1 个 slot。
        if count < 0:
            raise ValueError("count must be non-negative")
        if count > len(self._free):
            raise RuntimeError("KVPool exhausted")
        slots = [self._free.popleft() for _ in range(count)]
        self._allocated.update(slots)
        return slots

    def free_many(self, slots: list[int]) -> None:
        # 请求结束后释放它持有的所有 slot；未知 slot 会被忽略，便于幂等清理。
        for slot in slots:
            if slot in self._allocated:
                self._allocated.remove(slot)
                self._free.append(slot)

    @property
    def active_count(self) -> int:
        return len(self._allocated)


class ReqToTokenMap:
    # ReqToTokenMap 对应 SGLang 的 req_to_token：
    # (请求槽位 idx, 序列位置) -> KV slot。
    # 例如 req_pool_idx=3 的第 5 个 token 存在哪个 KV cache 位置。
    #
    #  用 ReqToTokenMap 表示就是：
    #
    #  (req_pool_idx, seq_pos) -> kv_slot
    #
    #  (0, 0) -> 7
    #  (0, 1) -> 8
    #  (0, 2) -> 9
    #  (0, 3) -> 10
    #
    #  (1, 0) -> 3
    #  (1, 1) -> 4
    #
    #  换成二维表更直观：
    #
    #  ReqToTokenMap
    #  =============
    #
    #                   seq_pos
    #  req_pool_idx      0     1     2     3
    #  ----------------------------------------
    #  0 / req-A         7     8     9     10
    #  1 / req-B         3     4     -     -
    #
    #  其中表里的数字就是 kv_slot。
    #
    #  和三个池子的关系：
    #
    #  ReqPool
    #    rid -> req_pool_idx
    #
    #  KVPool
    #    分配 kv_slot
    #
    #  ReqToTokenMap
    #    (req_pool_idx, seq_pos) -> kv_slot
    #
    #
    def __init__(self):
        self._map: dict[tuple[int, int], int] = {}

    def set(self, req_pool_idx: int, seq_pos: int, kv_slot: int) -> None:
        self._map[(req_pool_idx, seq_pos)] = kv_slot

    def get(self, req_pool_idx: int, seq_pos: int) -> int | None:
        return self._map.get((req_pool_idx, seq_pos))

    def remove_req(self, req_pool_idx: int) -> None:
        # 一个请求完成后，要删除它所有序列位置到 KV slot 的映射。
        for key in [key for key in self._map if key[0] == req_pool_idx]:
            del self._map[key]

    @property
    def size(self) -> int:
        return len(self._map)
