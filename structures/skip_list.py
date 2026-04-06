from __future__ import annotations
"""跳表 — O(log n) 查找/插入/删除 + 范围查询。
用于索引列的 BETWEEN 优化和 IndexScan 范围扫描。"""
import random
from typing import Any, List, Optional, Tuple

_MAX_LEVEL = 32


class _SkipNode:
    __slots__ = ('key', 'value', 'forward')
    def __init__(self, key: Any, value: Any,
                 level: int) -> None:
        self.key = key
        self.value = value
        self.forward: List[Optional[_SkipNode]] = [None] * (level + 1)


class SkipList:
    """概率平衡跳表。期望 O(log n) 操作。"""

    __slots__ = ('_header', '_level', '_size', '_p')

    def __init__(self, p: float = 0.5) -> None:
        self._header = _SkipNode(None, None, _MAX_LEVEL)
        self._level = 0
        self._size = 0
        self._p = p  # 层级提升概率

    def insert(self, key: Any, value: Any) -> None:
        """插入或更新。O(log n)。"""
        update = [None] * (_MAX_LEVEL + 1)
        current = self._header
        for i in range(self._level, -1, -1):
            while (current.forward[i] is not None
                   and current.forward[i].key < key):
                current = current.forward[i]
            update[i] = current
        current = current.forward[0]
        if current is not None and current.key == key:
            current.value = value
            return
        new_level = self._random_level()
        if new_level > self._level:
            for i in range(self._level + 1, new_level + 1):
                update[i] = self._header
            self._level = new_level
        node = _SkipNode(key, value, new_level)
        for i in range(new_level + 1):
            node.forward[i] = update[i].forward[i]
            update[i].forward[i] = node
        self._size += 1

    def search(self, key: Any) -> Optional[Any]:
        """点查。O(log n)。"""
        current = self._header
        for i in range(self._level, -1, -1):
            while (current.forward[i] is not None
                   and current.forward[i].key < key):
                current = current.forward[i]
        current = current.forward[0]
        if current is not None and current.key == key:
            return current.value
        return None

    def delete(self, key: Any) -> bool:
        """删除。O(log n)。"""
        update = [None] * (_MAX_LEVEL + 1)
        current = self._header
        for i in range(self._level, -1, -1):
            while (current.forward[i] is not None
                   and current.forward[i].key < key):
                current = current.forward[i]
            update[i] = current
        current = current.forward[0]
        if current is None or current.key != key:
            return False
        for i in range(self._level + 1):
            if update[i].forward[i] != current:
                break
            update[i].forward[i] = current.forward[i]
        while (self._level > 0
               and self._header.forward[self._level] is None):
            self._level -= 1
        self._size -= 1
        return True

    def range_query(self, lo: Any,
                    hi: Any) -> List[Tuple[Any, Any]]:
        """范围查询 [lo, hi]。O(log n + k)。"""
        results: list = []
        current = self._header
        for i in range(self._level, -1, -1):
            while (current.forward[i] is not None
                   and current.forward[i].key < lo):
                current = current.forward[i]
        current = current.forward[0]
        while current is not None and current.key <= hi:
            results.append((current.key, current.value))
            current = current.forward[0]
        return results

    def min(self) -> Optional[Tuple[Any, Any]]:
        """最小元素。O(1)。"""
        n = self._header.forward[0]
        return (n.key, n.value) if n else None

    def max(self) -> Optional[Tuple[Any, Any]]:
        """最大元素。O(log n)。"""
        current = self._header
        for i in range(self._level, -1, -1):
            while current.forward[i] is not None:
                current = current.forward[i]
        if current != self._header:
            return (current.key, current.value)
        return None

    @property
    def size(self) -> int:
        return self._size

    def _random_level(self) -> int:
        level = 0
        while random.random() < self._p and level < _MAX_LEVEL:
            level += 1
        return level
