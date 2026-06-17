"""Radix/prefix cache intuition demo.

Run from repo root:
    python sglang-learning-docs/06_demo_radix_cache.py
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Node:
    children: Dict[int, "Node"] = field(default_factory=dict)
    labels: List[str] = field(default_factory=list)


class PrefixTree:
    def __init__(self):
        self.root = Node()

    def insert(self, tokens: List[int], label: str):
        node = self.root
        for token in tokens:
            node = node.children.setdefault(token, Node())
        node.labels.append(label)

    def match_prefix(self, tokens: List[int]) -> List[int]:
        node = self.root
        matched = []
        for token in tokens:
            if token not in node.children:
                break
            matched.append(token)
            node = node.children[token]
        return matched

    def print_tree(self, node: Node | None = None, prefix: List[int] | None = None):
        node = self.root if node is None else node
        prefix = [] if prefix is None else prefix
        if node.labels:
            print(f"cached prefix={prefix}, labels={node.labels}")
        for token, child in sorted(node.children.items()):
            self.print_tree(child, prefix + [token])


def main():
    cache = PrefixTree()
    examples = [
        ([1, 2, 3, 4, 5], "req-1: Hello how are you"),
        ([1, 2, 3, 6, 7], "req-2: Hello how is it"),
        ([9, 8, 7], "req-3: unrelated"),
    ]

    print("insert cached prompts")
    for tokens, label in examples:
        print(f"  insert {tokens} -> {label}")
        cache.insert(tokens, label)

    print("\ncache tree leaves")
    cache.print_tree()

    probes = [
        [1, 2, 3, 4, 5, 99],
        [1, 2, 3, 6],
        [1, 2, 0],
        [9, 8, 7, 6],
    ]

    print("\nprefix match")
    for probe in probes:
        matched = cache.match_prefix(probe)
        print(f"  input={probe} matched={matched} miss_suffix={probe[len(matched):]}")


if __name__ == "__main__":
    main()

