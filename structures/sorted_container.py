from __future__ import annotations
"""块分解有序列表 — O(√n) 插入/删除，O(√n) 第 k 小查询。
用于 Mo's 算法的窗口状态维护（滑动中位数）。"""
import bisect
from typing import Any, List, Optional


class SortedList:
    """块分解有序列表。每块最多 LOAD_FACTOR 个元素。"""

    LOAD_FACTOR = 1000

    __slots__ = ('_blocks', '_block_sizes', '_total', '_maxes')

    def __init__(self) -> None:
        self._blocks: List[list] = [[]]
        self._block_sizes: List[int] = [0]
        self._maxes: List[Any] = []
        self._total = 0

    def add(self, value: Any) -> None:
        """O(√n) 有序插入。"""
        if self._total == 0:
            self._blocks[0].append(value)
            self._block_sizes[0] = 1
            self._maxes = [value]
            self._total = 1
            return
        block_idx = self._find_block(value)
        block = self._blocks[block_idx]
        bisect.insort(block, value)
        self._block_sizes[block_idx] += 1
        self._total += 1
        if block_idx < len(self._maxes):
            self._maxes[block_idx] = block[-1]
        else:
            self._maxes.append(block[-1])
        if self._block_sizes[block_idx] > 2 * self.LOAD_FACTOR:
            self._split(block_idx)

    def remove(self, value: Any) -> bool:
        """O(√n) 删除。"""
        if self._total == 0:
            return False
        block_idx = self._find_block(value)
        block = self._blocks[block_idx]
        i = bisect.bisect_left(block, value)
        if i < len(block) and block[i] == value:
            block.pop(i)
            self._block_sizes[block_idx] -= 1
            self._total -= 1
            if self._block_sizes[block_idx] == 0:
                if len(self._blocks) > 1:
                    self._blocks.pop(block_idx)
                    self._block_sizes.pop(block_idx)
                    self._maxes.pop(block_idx)
            elif block_idx < len(self._maxes):
                self._maxes[block_idx] = block[-1]
            return True
        return False

    def kth(self, k: int) -> Optional[Any]:
        """O(√n) 第 k 小（0-indexed）。"""
        if k < 0 or k >= self._total:
            return None
        remaining = k
        for bi, bs in enumerate(self._block_sizes):
            if remaining < bs:
                return self._blocks[bi][remaining]
            remaining -= bs
        return None

    def median(self) -> Optional[Any]:
        """中位数。"""
        if self._total == 0:
            return None
        if self._total % 2 == 1:
            return self.kth(self._total // 2)
        a = self.kth(self._total // 2 - 1)
        b = self.kth(self._total // 2)
        if a is not None and b is not None:
            return (a + b) / 2
        return a

    @property
    def size(self) -> int:
        return self._total

    def _find_block(self, value):
        if not self._maxes:
            return 0
        i = bisect.bisect_right(self._maxes, value)
        return min(i, len(self._blocks) - 1)

    def _split(self, block_idx):
        block = self._blocks[block_idx]
        mid = len(block) // 2
        left = block[:mid]
        right = block[mid:]
        self._blocks[block_idx] = left
        self._block_sizes[block_idx] = len(left)
        self._blocks.insert(block_idx + 1, right)
        self._block_sizes.insert(block_idx + 1, len(right))
        self._maxes[block_idx] = left[-1] if left else self._maxes[block_idx]
        self._maxes.insert(block_idx + 1, right[-1] if right else self._maxes[block_idx])
