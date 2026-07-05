from __future__ import annotations

import time
from dataclasses import dataclass, field


def _common_prefix_len(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    limit = min(len(left), len(right))
    for index in range(limit):
        if left[index] != right[index]:
            return index
    return limit


@dataclass
class RadixNode:
    # A compressed radix edge: key_segment[i] is backed by slot_segment[i].
    key_segment: tuple[int, ...]
    slot_segment: tuple[int, ...]
    parent: RadixNode | None = None
    children: dict[int, RadixNode] = field(default_factory=dict)
    last_access_time: float = field(default_factory=time.monotonic)


@dataclass(frozen=True)
class PrefixMatch:
    token_count: int
    slot_ids: tuple[int, ...]
    last_node: RadixNode


@dataclass(frozen=True)
class InsertResult:
    prefix_len: int
    total_len: int
    inserted_slots: tuple[int, ...]


class MiniRadixCache:
    # A small, pure-Python radix cache for teaching prefix KV reuse.
    # It indexes token prefixes and stores the KV slot id for each cached token.
    def __init__(self):
        self.root = RadixNode(key_segment=(), slot_segment=())

    def match_prefix(self, token_ids: list[int] | tuple[int, ...]) -> PrefixMatch:
        key = tuple(int(token_id) for token_id in token_ids)
        node = self.root
        matched_slots: list[int] = []
        remaining = key
        access_time = time.monotonic()

        while remaining:
            child = node.children.get(remaining[0])
            if child is None:
                break

            child.last_access_time = access_time
            prefix_len = _common_prefix_len(child.key_segment, remaining)
            if prefix_len == 0:
                break

            if prefix_len < len(child.key_segment):
                node = self._split_node(child, prefix_len)
                matched_slots.extend(node.slot_segment)
                break

            matched_slots.extend(child.slot_segment)
            node = child
            remaining = remaining[prefix_len:]

        node.last_access_time = access_time
        return PrefixMatch(
            token_count=len(matched_slots),
            slot_ids=tuple(matched_slots),
            last_node=node,
        )

    def insert(
        self,
        token_ids: list[int] | tuple[int, ...],
        slot_ids: list[int] | tuple[int, ...],
    ) -> InsertResult:
        key = tuple(int(token_id) for token_id in token_ids)
        slots = tuple(int(slot_id) for slot_id in slot_ids)
        if len(key) != len(slots):
            raise ValueError("token_ids and slot_ids must have the same length")
        if not key:
            return InsertResult(prefix_len=0, total_len=0, inserted_slots=())

        node = self.root
        remaining_key = key
        remaining_slots = slots
        prefix_len_total = 0
        access_time = time.monotonic()

        while remaining_key:
            child = node.children.get(remaining_key[0])
            if child is None:
                self._add_child(node, remaining_key, remaining_slots, access_time)
                return InsertResult(
                    prefix_len=prefix_len_total,
                    total_len=len(key),
                    inserted_slots=remaining_slots,
                )

            child.last_access_time = access_time
            prefix_len = _common_prefix_len(child.key_segment, remaining_key)
            if prefix_len == 0:
                self._add_child(node, remaining_key, remaining_slots, access_time)
                return InsertResult(
                    prefix_len=prefix_len_total,
                    total_len=len(key),
                    inserted_slots=remaining_slots,
                )

            prefix_len_total += prefix_len
            remaining_key = remaining_key[prefix_len:]
            remaining_slots = remaining_slots[prefix_len:]

            if prefix_len < len(child.key_segment):
                node = self._split_node(child, prefix_len)
                if remaining_key:
                    self._add_child(node, remaining_key, remaining_slots, access_time)
                    return InsertResult(
                        prefix_len=prefix_len_total,
                        total_len=len(key),
                        inserted_slots=remaining_slots,
                    )
                return InsertResult(
                    prefix_len=prefix_len_total,
                    total_len=len(key),
                    inserted_slots=(),
                )

            node = child

        return InsertResult(
            prefix_len=prefix_len_total,
            total_len=len(key),
            inserted_slots=(),
        )

    def reset(self) -> None:
        self.root.children.clear()

    def total_size(self) -> int:
        total = 0
        stack = list(self.root.children.values())
        while stack:
            node = stack.pop()
            total += len(node.slot_segment)
            stack.extend(node.children.values())
        return total

    def _add_child(
        self,
        parent: RadixNode,
        key_segment: tuple[int, ...],
        slot_segment: tuple[int, ...],
        access_time: float,
    ) -> RadixNode:
        child = RadixNode(
            key_segment=key_segment,
            slot_segment=slot_segment,
            parent=parent,
            last_access_time=access_time,
        )
        parent.children[key_segment[0]] = child
        return child

    def _split_node(self, child: RadixNode, split_len: int) -> RadixNode:
        if split_len <= 0 or split_len >= len(child.key_segment):
            raise ValueError("split_len must split the child node")

        parent = child.parent
        if parent is None:
            raise RuntimeError("cannot split root node")

        prefix_key = child.key_segment[:split_len]
        prefix_slots = child.slot_segment[:split_len]
        suffix_key = child.key_segment[split_len:]
        suffix_slots = child.slot_segment[split_len:]

        split_node = RadixNode(
            key_segment=prefix_key,
            slot_segment=prefix_slots,
            parent=parent,
            last_access_time=child.last_access_time,
        )
        split_node.children[suffix_key[0]] = child
        parent.children[prefix_key[0]] = split_node

        child.key_segment = suffix_key
        child.slot_segment = suffix_slots
        child.parent = split_node
        return split_node
