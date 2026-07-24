from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass, field


def _common_prefix_len(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    # 返回两段 token 从头开始连续相同的长度。
    #
    # 例子：
    #   left  = (1, 2, 3)
    #   right = (1, 2, 9)
    #   返回 2
    #
    # radix cache 的核心操作就是找“公共前缀”：公共前缀已经缓存，可以复用；
    # 第一个不同 token 之后的部分，才需要新分配 KV slot 或产生分叉。
    limit = min(len(left), len(right))
    for index in range(limit):
        if left[index] != right[index]:
            return index
    return limit


@dataclass
class RadixNode:
    # 压缩 radix 树的一个节点：key_segment[i] 对应 slot_segment[i]。
    # 多个连续 token 会被压在同一条边上，只有出现分叉时才拆节点。
    #
    # 普通 trie 可能长这样：
    #   root -> 1 -> 2 -> 3
    #
    # 压缩 radix tree 会把没有分叉的连续路径合并：
    #   root -> (1, 2, 3)
    #
    # 在 KV cache 里：
    #   key_segment  = token id 片段
    #   slot_segment = 这些 token 对应的 KV cache slot id 片段
    #
    # children 的 key 不是完整路径，而是子边的第一个 token：
    #   parent.children[child.key_segment[0]] = child
    # 这样查下一条边时，只需要看 remaining_key[0]。
    key_segment: tuple[int, ...]
    slot_segment: tuple[int, ...]
    parent: RadixNode | None = None
    children: dict[int, RadixNode] = field(default_factory=dict)
    # 这里记录访问时间，LRU 淘汰时会优先删除最久没有访问的叶子节点。
    last_access_time: float = field(default_factory=time.monotonic)
    # ref_count 表示当前有多少活跃请求正在借用这个节点的 KV slot。
    # LRU 淘汰只能删除 ref_count == 0 的节点，避免把正在被请求使用的 slot 释放掉。
    ref_count: int = 0


@dataclass(frozen=True)
class PrefixMatch:
    # token_count 是命中的 token 数；slot_ids 是这些 token 已缓存的 KV slot。
    token_count: int
    slot_ids: tuple[int, ...]
    # last_node 便于后续扩展为从命中位置继续插入；当前教学版主要用于观察。
    last_node: RadixNode
    # matched_nodes 是本次命中的 radix 节点列表。
    # scheduler 复用 prefix 时会 pin 这些节点，请求结束后再 release。
    matched_nodes: tuple[RadixNode, ...] = ()


@dataclass(frozen=True)
class InsertResult:
    # prefix_len 是已有 cache 覆盖的长度，inserted_slots 是这次新交给 cache 托管的 slot。
    #
    # 例子：
    #   已缓存 token: [1, 2, 3] -> slot [10, 11, 12]
    #   新插入 token: [1, 2, 4] -> slot [20, 21, 22]
    #
    #   prefix_len     = 2       # [1, 2] 已经存在
    #   total_len      = 3       # 新请求总长度
    #   inserted_slots = (22,)   # 只有 token 4 对应的 slot 新交给 cache
    prefix_len: int
    total_len: int
    inserted_slots: tuple[int, ...]
    # evicted_slots 是因为容量上限被 LRU 淘汰的 slot。
    # radix_cache 只决定“哪些 slot 不再归 cache 管”，真正释放 KVPool 要由 scheduler 做。
    evicted_slots: tuple[int, ...] = ()


class MiniRadixCache:
    # 纯 Python 教学版 radix cache：按 token 前缀索引 KV slot，实现 prefix 复用。
    # 它只管理 slot id 的归属，不保存真实 KV 张量。
    #
    # 直觉：
    #   prompt A: [1, 2, 3]
    #   prompt B: [1, 2, 4]
    #
    # 两个 prompt 的 [1, 2] 前缀相同，B 不需要重复计算这部分 KV，
    # 只要从 radix cache 找到 [1, 2] 对应的 slot 即可。
    #
    # 树形结构大致是：
    #   root
    #    └── (1, 2)
    #         ├── (3)
    #         └── (4)
    def __init__(self, max_slots: int | None = None, *, page_size: int = 1):
        if max_slots is not None and max_slots <= 0:
            raise ValueError("max_slots must be positive")
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        # max_slots 是 radix cache 最多托管多少个 KV slot。
        # None 表示不限制容量；设置为整数后，insert 会在末尾触发 LRU 淘汰。
        self.max_slots = max_slots
        self.page_size = page_size
        # root 是哨兵节点，不代表真实 token，也没有 slot。
        self.root = RadixNode(key_segment=(), slot_segment=())

    def match_prefix(
        self,
        token_ids: list[int] | tuple[int, ...],
        *,
        pin: bool = False,
    ) -> PrefixMatch:
        # 命中、pin 和 suffix 分配见 my-sglang/docs/dynamic-flows.md#radix-flow。
        # 统一转成 tuple，后面切片、比较、作为不可变片段保存都更简单。
        aligned_len = len(token_ids) // self.page_size * self.page_size
        key = tuple(int(token_id) for token_id in token_ids[:aligned_len])
        node = self.root
        matched_slots: list[int] = []
        matched_nodes: list[RadixNode] = []
        # remaining 表示“从当前 node 往下，还没有匹配的 token 后缀”。
        remaining = key
        access_time = time.monotonic()

        while remaining:
            # children 用“子边第一个 token”做索引。
            # 所以要走下一条边，只看 remaining[0]。
            #
            # 例子：
            #   remaining = (1, 2, 9)
            #   node.children.get(1) 能直接找到所有以 1 开头的压缩边。
            child = node.children.get(remaining[0])
            if child is None:
                # 当前节点下没有以 remaining[0] 开头的边，匹配停止。
                break

            child.last_access_time = access_time
            prefix_len = _common_prefix_len(child.key_segment, remaining)
            if prefix_len == 0:
                # 理论上 children.get(remaining[0]) 已经保证第一个 token 相同；
                # 这里保留防御判断，避免树结构被外部错误修改后继续产生错误结果。
                break

            if prefix_len < len(child.key_segment):
                # 查询落在压缩边中间：拆出公共前缀节点，命中到这里为止。
                #
                # 例子：
                #   树里已有边: (1, 2, 3)
                #   查询 key:  (1, 2, 9)
                #
                # 拆之前：
                #   root -> (1, 2, 3)
                #
                # 拆之后：
                #   root -> (1, 2) -> (3)
                #
                # 查询只能命中 (1, 2)，不能继续走到 (3)。
                node = self._split_node(child, prefix_len)
                matched_slots.extend(node.slot_segment)
                matched_nodes.append(node)
                break

            # 完整吃掉当前压缩边，继续向下匹配剩余 token。
            matched_slots.extend(child.slot_segment)
            matched_nodes.append(child)
            node = child
            remaining = remaining[prefix_len:]

        node.last_access_time = access_time
        if pin and matched_slots:
            self.inc_lock_ref(node)
        return PrefixMatch(
            token_count=len(matched_slots),
            slot_ids=tuple(matched_slots),
            last_node=node,
            matched_nodes=tuple(matched_nodes),
        )

    def pin_nodes(self, nodes: list[RadixNode] | tuple[RadixNode, ...]) -> None:
        # pin = 告诉 cache：这些节点的 slot 正在被活跃请求借用，暂时不能淘汰。
        #
        # 例子：
        #   请求 B 命中了请求 A 留下的 prefix [1, 2]。
        #   B 运行期间，[1, 2] 对应的 KV slot 仍然要给模型用。
        #   这时即使 cache 满了，也不能把 [1, 2] 淘汰掉。
        if nodes:
            self.inc_lock_ref(nodes[-1])

    def release_nodes(self, nodes: list[RadixNode] | tuple[RadixNode, ...]) -> None:
        # release = 请求结束，不再借用这些 prefix slot。
        # release 后节点重新成为 LRU 淘汰候选。
        if nodes:
            self.dec_lock_ref(nodes[-1])

    def inc_lock_ref(self, node: RadixNode) -> None:
        # 与真实 SGLang 一样，锁住终点意味着整条祖先路径都不能被淘汰。
        while node is not self.root:
            node.ref_count += 1
            if node.parent is None:
                raise RuntimeError("radix node is detached from this tree")
            node = node.parent

    def dec_lock_ref(self, node: RadixNode) -> None:
        while node is not self.root:
            if node.ref_count <= 0:
                raise RuntimeError("radix cache node released more times than pinned")
            node.ref_count -= 1
            if node.parent is None:
                raise RuntimeError("radix node is detached from this tree")
            node = node.parent

    def insert(
        self,
        token_ids: Iterable[int],
        slot_ids: Iterable[int],
    ) -> InsertResult:
        # 插入后的 slot 归属变化见 my-sglang/docs/dynamic-flows.md#radix-flow。
        # key 和 slots 一一对应：
        #   key[i]   是第 i 个 token id
        #   slots[i] 是这个 token 的 KV cache slot id
        raw_key = tuple(int(token_id) for token_id in token_ids)
        raw_slots = tuple(int(slot_id) for slot_id in slot_ids)
        if len(raw_key) != len(raw_slots):
            raise ValueError("token_ids and slot_ids must have the same length")
        aligned_len = len(raw_key) // self.page_size * self.page_size
        key = raw_key[:aligned_len]
        slots = raw_slots[:aligned_len]
        if not key:
            # 空 prompt 没有 token，也没有 KV slot 可以缓存。
            return InsertResult(prefix_len=0, total_len=0, inserted_slots=())

        # node 就是一个游标的概念
        node = self.root

        # remaining_key / remaining_slots 始终保持对齐。
        # 每匹配掉 prefix_len 个 token，就同步丢掉 prefix_len 个 slot。
        remaining_key = key
        remaining_slots = slots
        # 已经被旧 cache 覆盖的 token 数。
        prefix_len_total = 0
        access_time = time.monotonic()

        while remaining_key:
            # radix tree 的每个节点用“下一段压缩边的第一个 token”分流。
            # 因此插入时先看 remaining_key[0]，判断是否存在共享前缀的候选边。
            child = node.children.get(remaining_key[0])
            if child is None:
                # 没有共享前缀，剩余 token 作为一条新的压缩边挂到当前节点。
                #
                # 例子：
                #   当前 node 下已有 child: (1, 2)
                #   要插入 remaining:       (9, 8)
                #
                # 因为没有 children[9]，直接新增：
                #   node -> (9, 8)
                self._add_child(node, remaining_key, remaining_slots, access_time)
                evicted_slots = self._evict_lru_if_needed()
                return InsertResult(
                    prefix_len=prefix_len_total,
                    total_len=len(key),
                    inserted_slots=remaining_slots,
                    evicted_slots=evicted_slots,
                )

            child.last_access_time = access_time
            prefix_len = _common_prefix_len(child.key_segment, remaining_key)
            if prefix_len == 0:
                # 正常情况下不会发生，因为 child 是按 remaining_key[0] 找到的；
                # 保留分支是为了让 insert 在树结构异常时仍能退化为新增边。
                self._add_child(node, remaining_key, remaining_slots, access_time)
                evicted_slots = self._evict_lru_if_needed()
                return InsertResult(
                    prefix_len=prefix_len_total,
                    total_len=len(key),
                    inserted_slots=remaining_slots,
                    evicted_slots=evicted_slots,
                )

            prefix_len_total += prefix_len
            remaining_key = remaining_key[prefix_len:]
            remaining_slots = remaining_slots[prefix_len:]

            if prefix_len < len(child.key_segment):
                # 插入路径在压缩边中间分叉：先拆边，再把剩余 suffix 接到拆出的节点下。
                #
                # 例子：
                #   树里已有: (1, 2, 3)
                #   新插入:   (1, 2, 4)
                #
                # 公共前缀是 (1, 2)，先拆：
                #   root -> (1, 2) -> (3)
                #
                # 再把新 suffix (4) 挂上去：
                #   root -> (1, 2)
                #            ├── (3)
                #            └── (4)
                #
                # node 就是split后的公共前缀
                node = self._split_node(child, prefix_len)

                # 新插入的 key 除了公共前缀之外，还有没有后缀；有后缀就新增分支，没有后缀就不用加节点
                if remaining_key:
                    self._add_child(node, remaining_key, remaining_slots, access_time)
                    evicted_slots = self._evict_lru_if_needed()
                    return InsertResult(
                        prefix_len=prefix_len_total,
                        total_len=len(key),
                        inserted_slots=remaining_slots,
                        evicted_slots=evicted_slots,
                    )
                evicted_slots = self._evict_lru_if_needed()
                return InsertResult(
                    prefix_len=prefix_len_total,
                    total_len=len(key),
                    inserted_slots=(),
                    evicted_slots=evicted_slots,
                )

            # child.key_segment 被完整匹配，继续向 child 的子树插入剩余后缀。
            node = child

        # 走到这里表示整条 key 都已经在 cache 里，没有新增 slot。
        return InsertResult(
            prefix_len=prefix_len_total,
            total_len=len(key),
            inserted_slots=(),
            evicted_slots=self._evict_lru_if_needed(),
        )

    def reset(self) -> None:
        # 清空所有真实 token 节点，保留 root 哨兵节点。
        self.root.children.clear()

    def total_size(self) -> int:
        # 统计当前 radix cache 托管了多少个 KV slot。
        # 因为每个 token 对应一个 slot，所以累加所有 node.slot_segment 长度即可。
        total = 0
        stack = list(self.root.children.values())
        while stack:
            node = stack.pop()
            total += len(node.slot_segment)
            stack.extend(node.children.values())
        return total

    def evictable_size(self) -> int:
        return self._size_by_lock(locked=False)

    def protected_size(self) -> int:
        return self._size_by_lock(locked=True)

    def _size_by_lock(self, *, locked: bool) -> int:
        total = 0
        stack = list(self.root.children.values())
        while stack:
            node = stack.pop()
            if (node.ref_count > 0) is locked:
                total += len(node.slot_segment)
            stack.extend(node.children.values())
        return total

    def evict(self, num_slots: int) -> tuple[int, ...]:
        if num_slots <= 0:
            return ()
        evicted: list[int] = []
        while len(evicted) < num_slots:
            victim = self._find_lru_evictable_leaf()
            if victim is None:
                break
            evicted.extend(self._remove_leaf(victim))
        return tuple(evicted)

    def _add_child(
        self,
        parent: RadixNode,
        key_segment: tuple[int, ...],
        slot_segment: tuple[int, ...],
        access_time: float,
    ) -> RadixNode:
        # 新增一条压缩边。调用方必须保证 key_segment 非空，
        # 因为 parent.children 需要用 key_segment[0] 当索引。
        child = RadixNode(
            key_segment=key_segment,
            slot_segment=slot_segment,
            parent=parent,
            last_access_time=access_time,
        )
        # 关键约定：
        #   children[token] = 以 token 开头的那条压缩边
        #
        # 这就是 match/insert 里用 remaining_key[0] 查 child 的原因。
        parent.children[key_segment[0]] = child
        return child

    def _split_node(self, child: RadixNode, split_len: int) -> RadixNode:
        # split 前后树形图见 my-sglang/docs/data-structures.md#radix-tree。
        # 把一条压缩边从中间拆开。
        #
        # 拆之前：
        #   parent -> child(key=(1, 2, 3), slots=(10, 11, 12))
        #
        # split_len = 2
        #
        # 拆之后：
        #   parent -> split_node(key=(1, 2), slots=(10, 11))
        #              └── child(key=(3), slots=(12))
        if split_len <= 0 or split_len >= len(child.key_segment):
            raise ValueError("split_len must split the child node")

        parent = child.parent
        if parent is None:
            raise RuntimeError("cannot split root node")

        # Python 切片规则：
        #   xs[:n]  取下标 0 到 n-1，不包含 n
        #   xs[n:]  取下标 n 到最后
        #
        # 例子：
        #   child.key_segment  = (1, 2, 3)
        #   child.slot_segment = (10, 11, 12)
        #   split_len = 2
        #
        # 切完后：
        #   prefix_key   = (1, 2)      # child.key_segment[:2]
        #   prefix_slots = (10, 11)    # child.slot_segment[:2]
        #   suffix_key   = (3,)        # child.key_segment[2:]
        #   suffix_slots = (12,)       # child.slot_segment[2:]
        #
        # 注意 Python 里单元素 tuple 要写成 (3,)，不能写成 (3)；
        # (3) 只是整数 3 加了一层括号。
        prefix_key = child.key_segment[:split_len]
        prefix_slots = child.slot_segment[:split_len]
        suffix_key = child.key_segment[split_len:]
        suffix_slots = child.slot_segment[split_len:]

        # parent -> child 变成 parent -> split_node -> child。
        # split_node 持有公共前缀，原 child 缩短为剩余 suffix。
        #
        # 注意 slot 也必须同步拆分，保证：
        #   key_segment[i] 仍然对应 slot_segment[i]
        split_node = RadixNode(
            key_segment=prefix_key,
            slot_segment=prefix_slots,
            parent=parent,
            last_access_time=child.last_access_time,
            # child 原本被锁时，新插入的祖先也必须拥有相同 lock ref；
            # 之后从 child dec_lock_ref 会沿新父节点一起释放。
            ref_count=child.ref_count,
        )
        # children 字典的约定：
        #   children[某条子边的第一个 token] = 那个子节点
        #
        # 为什么用 [0]：
        #   key_segment 是一整段压缩边，比如 (1, 2) 或 (3,)
        #   但查找下一条边时，只需要用“第一个 token”分流。
        #
        # 沿用上面的例子：
        #   prefix_key = (1, 2)
        #   suffix_key = (3,)
        #
        # 拆之前：
        #   parent.children[1] -> child(key=(1, 2, 3))
        #
        # 拆之后需要变成：
        #   parent.children[1]      -> split_node(key=(1, 2))
        #   split_node.children[3]  -> child(key=(3,))
        #
        # 所以这里是：
        #   suffix_key[0] = 3，用来把原 child 挂到 split_node 下面
        #   prefix_key[0] = 1，用来把 split_node 挂回 parent 下面
        split_node.children[suffix_key[0]] = child
        parent.children[prefix_key[0]] = split_node

        child.key_segment = suffix_key
        child.slot_segment = suffix_slots
        child.parent = split_node
        return split_node

    def _evict_lru_if_needed(self) -> tuple[int, ...]:
        # LRU 与 pin 的配合见 my-sglang/docs/data-structures.md#radix-lru。
        # 没有容量上限时，不做淘汰。
        if self.max_slots is None:
            return ()

        evicted_slots: list[int] = []
        # total_size() 超过 max_slots，就持续删最久未访问的“未 pin 叶子节点”。
        #
        # 为什么只淘汰叶子：
        #   删除内部节点会连带删掉整棵子树，容易误伤仍然有用的后缀。
        #   教学版先用最直观、最安全的粒度：叶子节点。
        #
        # 为什么要求 ref_count == 0：
        #   ref_count > 0 表示有活跃请求正在借用这个节点的 KV slot。
        while self.total_size() > self.max_slots:
            evicted_slots.extend(self.evict(self.total_size() - self.max_slots))
            break
        return tuple(evicted_slots)

    def _find_lru_evictable_leaf(self) -> RadixNode | None:
        # LRU = Least Recently Used，最久没访问的节点优先淘汰。
        # 这里遍历所有叶子，选 last_access_time 最小的那个。
        victim: RadixNode | None = None
        stack = list(self.root.children.values())
        while stack:
            node = stack.pop()
            stack.extend(node.children.values())
            if node.children:
                continue
            if node.ref_count > 0:
                continue
            if victim is None or node.last_access_time < victim.last_access_time:
                victim = node
        return victim

    def _remove_leaf(self, node: RadixNode) -> tuple[int, ...]:
        # 从树上删除一个叶子节点，并把它托管的 slot 返回给调用方。
        # radix_cache 自己不碰 KVPool，因为它不知道真实 KV 张量存在哪里。
        if node.children:
            raise ValueError("only leaf nodes can be removed")
        if node.ref_count > 0:
            raise ValueError("cannot remove a pinned radix cache node")
        parent = node.parent
        if parent is None:
            raise RuntimeError("cannot remove root node")

        removed_slots = node.slot_segment
        del parent.children[node.key_segment[0]]
        node.parent = None
        return removed_slots
