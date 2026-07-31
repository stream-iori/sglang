"""按 token 前缀复用 KV slot 的压缩 radix tree。

每个节点同时保存等长的 ``key_segment``（token）和 ``slot_segment``（KV
位置）。树负责前缀所有权、pin 和 LRU 选择；真正释放 slot 仍由 allocator 做。
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass, field


def _common_prefix_len(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    """返回两个 tuple 从头连续相同的元素数量。"""
    limit = min(len(left), len(right))
    for index in range(limit):
        if left[index] != right[index]:
            return index
    return limit


@dataclass
class RadixNode:
    """压缩树的一条边；连续且无分叉的 token 会合并在同一节点。"""

    key_segment: tuple[int, ...]  # 这一段 token id。
    slot_segment: tuple[int, ...]  # 与 token 一一对应的 KV slot。
    parent: RadixNode | None = None  # root 的 parent 为 None。
    children: dict[int, RadixNode] = field(default_factory=dict)  # 首 token -> 子边。
    last_access_time: float = field(default_factory=time.monotonic)  # LRU 时间。
    ref_count: int = 0  # 活跃请求引用数；大于 0 时不能淘汰。


@dataclass(frozen=True)
class PrefixMatch:
    """一次前缀查询的只读结果。"""

    token_count: int  # 命中的 token 数。
    slot_ids: tuple[int, ...]  # 命中 token 对应的 KV slot。
    last_node: RadixNode  # 命中路径的终点。
    matched_nodes: tuple[RadixNode, ...] = ()  # 依次经过的真实节点。


@dataclass(frozen=True)
class InsertResult:
    """插入后的所有权变化；释放 ``evicted_slots`` 由 scheduler 负责。"""

    prefix_len: int  # 已经被旧 cache 覆盖的 token 数。
    total_len: int  # page 对齐后的输入 token 总数。
    inserted_slots: tuple[int, ...]  # 新交给 cache 托管的 slot。
    evicted_slots: tuple[int, ...] = ()  # 因容量上限被淘汰的 slot。


class MiniRadixCache:
    """纯 Python、page-aware 的教学 radix KV cache。

    例如 ``[1, 2, 3]`` 和 ``[1, 2, 4]`` 会形成：

    ``root -> (1, 2) -> (3) / (4)``
    """
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
        """查找最长的 page-aligned 前缀；``pin=True`` 时锁住命中路径。"""

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
        """锁住一条已匹配路径，使其暂时不能被 LRU 淘汰。"""
        # pin = 告诉 cache：这些节点的 slot 正在被活跃请求借用，暂时不能淘汰。
        #
        # 例子：
        #   请求 B 命中了请求 A 留下的 prefix [1, 2]。
        #   B 运行期间，[1, 2] 对应的 KV slot 仍然要给模型用。
        #   这时即使 cache 满了，也不能把 [1, 2] 淘汰掉。
        if nodes:
            self.inc_lock_ref(nodes[-1])

    def release_nodes(self, nodes: list[RadixNode] | tuple[RadixNode, ...]) -> None:
        """释放一次路径引用，使其重新具备可淘汰资格。"""
        # release = 请求结束，不再借用这些 prefix slot。
        # release 后节点重新成为 LRU 淘汰候选。
        if nodes:
            self.dec_lock_ref(nodes[-1])

    def inc_lock_ref(self, node: RadixNode) -> None:
        """从终点到 root 依次增加引用计数。"""
        # 与真实 SGLang 一样，锁住终点意味着整条祖先路径都不能被淘汰。
        while node is not self.root:
            node.ref_count += 1
            if node.parent is None:
                raise RuntimeError("radix node is detached from this tree")
            node = node.parent

    def dec_lock_ref(self, node: RadixNode) -> None:
        """从终点到 root 依次减少引用计数。"""
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
        """插入一条 ``token -> slot`` 路径，并返回 cache 所有权变化。"""

        tokens = tuple(int(token_id) for token_id in token_ids)
        slots = tuple(int(slot_id) for slot_id in slot_ids)
        if len(tokens) != len(slots):
            raise ValueError("token_ids and slot_ids must have the same length")

        # cache 只接管完整 page，尾部不完整 page 仍归活跃请求所有。
        aligned_len = len(tokens) // self.page_size * self.page_size
        tokens = tokens[:aligned_len]
        slots = slots[:aligned_len]
        if not tokens:
            return InsertResult(prefix_len=0, total_len=0, inserted_slots=())

        cursor = self.root
        remaining_tokens = tokens
        remaining_slots = slots
        matched_prefix_len = 0
        access_time = time.monotonic()

        while remaining_tokens:
            # children 以压缩边的第一个 token 为索引。
            child = cursor.children.get(remaining_tokens[0])
            if child is None:
                # 没有共享前缀：把全部剩余内容压成一条新边。
                self._add_child(
                    cursor, remaining_tokens, remaining_slots, access_time
                )
                return self._finish_insert(
                    matched_prefix_len=matched_prefix_len,
                    total_len=len(tokens),
                    inserted_slots=remaining_slots,
                )

            child.last_access_time = access_time
            shared_len = _common_prefix_len(
                child.key_segment, remaining_tokens
            )
            if shared_len == 0:
                # 防御分支：树索引异常时退化为新增边，而不是继续错误匹配。
                self._add_child(
                    cursor, remaining_tokens, remaining_slots, access_time
                )
                return self._finish_insert(
                    matched_prefix_len=matched_prefix_len,
                    total_len=len(tokens),
                    inserted_slots=remaining_slots,
                )

            matched_prefix_len += shared_len
            remaining_tokens = remaining_tokens[shared_len:]
            remaining_slots = remaining_slots[shared_len:]

            if shared_len < len(child.key_segment):
                # 在压缩边中间分叉：公共前缀成为父节点，新旧 suffix 各走一边。
                split_node = self._split_node(child, shared_len)
                if remaining_tokens:
                    self._add_child(
                        split_node,
                        remaining_tokens,
                        remaining_slots,
                        access_time,
                    )
                return self._finish_insert(
                    matched_prefix_len=matched_prefix_len,
                    total_len=len(tokens),
                    inserted_slots=remaining_slots,
                )

            # 当前压缩边被完整匹配，继续向下寻找剩余 suffix。
            cursor = child

        # 所有 token 都已存在，没有新的 cache 所有权。
        return self._finish_insert(
            matched_prefix_len=matched_prefix_len,
            total_len=len(tokens),
            inserted_slots=(),
        )

    def _finish_insert(
        self,
        *,
        matched_prefix_len: int,
        total_len: int,
        inserted_slots: tuple[int, ...],
    ) -> InsertResult:
        """统一执行容量淘汰并构造插入结果。"""

        return InsertResult(
            prefix_len=matched_prefix_len,
            total_len=total_len,
            inserted_slots=inserted_slots,
            evicted_slots=self._evict_lru_if_needed(),
        )

    def reset(self) -> None:
        """清空所有真实 token 节点，保留 root 哨兵。"""
        self.root.children.clear()

    def total_size(self) -> int:
        """统计 cache 当前托管的 KV slot 总数。"""
        total = 0
        stack = list(self.root.children.values())
        while stack:
            node = stack.pop()
            total += len(node.slot_segment)
            stack.extend(node.children.values())
        return total

    def evictable_size(self) -> int:
        """统计未被活跃请求引用的 slot 数。"""
        return self._size_by_lock(locked=False)

    def protected_size(self) -> int:
        """统计被活跃请求引用、不可淘汰的 slot 数。"""
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
        """按 LRU 依次删除未锁定叶子，直到达到目标或没有候选。"""
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
        """在 ``parent`` 下新增一条非空压缩边。"""
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
        #   parent
        #   └── children
        #       ├── 10 ──────> child_a
        #       └── 20 ──────> child_b
        parent.children[key_segment[0]] = child
        return child

    def _split_node(self, child: RadixNode, split_len: int) -> RadixNode:
        """将一条压缩边拆成公共 prefix 节点和原 suffix 节点。

        ``parent -> (1,2,3)`` 在 ``split_len=2`` 时变为
        ``parent -> (1,2) -> (3)``；token 与 slot 必须同步切分。
        """
        if split_len <= 0 or split_len >= len(child.key_segment):
            raise ValueError("split_len must split the child node")

        parent = child.parent
        if parent is None:
            raise RuntimeError("cannot split root node")

        prefix_key = child.key_segment[:split_len]
        prefix_slots = child.slot_segment[:split_len]
        suffix_key = child.key_segment[split_len:]
        suffix_slots = child.slot_segment[split_len:]

        # parent -> child 变成 parent -> split_node -> child。
        split_node = RadixNode(
            key_segment=prefix_key,
            slot_segment=prefix_slots,
            parent=parent,
            last_access_time=child.last_access_time,
            # child 原本被锁时，新插入的祖先也必须拥有相同 lock ref；
            # 之后从 child dec_lock_ref 会沿新父节点一起释放。
            ref_count=child.ref_count,
        )
        # children 始终用压缩边的首 token 建索引。
        split_node.children[suffix_key[0]] = child
        parent.children[prefix_key[0]] = split_node

        child.key_segment = suffix_key
        child.slot_segment = suffix_slots
        child.parent = split_node
        return split_node

    def _evict_lru_if_needed(self) -> tuple[int, ...]:
        """容量超限时淘汰最少数量的未锁定叶子。"""

        if self.max_slots is None:
            return ()

        overflow = self.total_size() - self.max_slots
        return self.evict(overflow) if overflow > 0 else ()

    def _find_lru_evictable_leaf(self) -> RadixNode | None:
        """遍历所有未锁定叶子，返回最久未访问的节点。"""
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
        """删除一个未锁定叶子，并将其 slot 返回给调用方释放。"""
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
