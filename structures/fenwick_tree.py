from __future__ import annotations
"""Fenwick Tree (BIT) — prefix sum O(log n), point update O(log n).
Lower constant than Segment Tree. Used for window running sums."""
from typing import List


class FenwickTree:
    """Binary Indexed Tree for prefix sums."""

    __slots__ = ('_n', '_tree')

    def __init__(self, n: int) -> None:
        self._n = n
        self._tree = [0] * (n + 1)

    @staticmethod
    def from_list(data: list) -> FenwickTree:
        """Build from list in O(n)."""
        ft = FenwickTree(len(data))
        for i, v in enumerate(data):
            ft._tree[i + 1] = v
        for i in range(1, len(data) + 1):
            j = i + (i & (-i))
            if j <= len(data):
                ft._tree[j] += ft._tree[i]
        return ft

    def update(self, i: int, delta: int) -> None:
        """Add delta to position i. O(log n)."""
        i += 1
        while i <= self._n:
            self._tree[i] += delta
            i += i & (-i)

    def prefix_sum(self, i: int) -> int:
        """Sum of [0, i] inclusive. O(log n)."""
        s = 0
        i += 1
        while i > 0:
            s += self._tree[i]
            i -= i & (-i)
        return s

    def range_sum(self, l: int, r: int) -> int:
        """Sum of [l, r] inclusive. O(log n)."""
        if l > r:
            return 0
        s = self.prefix_sum(r)
        if l > 0:
            s -= self.prefix_sum(l - 1)
        return s

    def find_kth(self, k: int) -> int:
        """Find smallest i such that prefix_sum(i) >= k. O(log n).
        Assumes all values are non-negative."""
        pos = 0
        bit_mask = 1
        while bit_mask <= self._n:
            bit_mask <<= 1
        bit_mask >>= 1
        while bit_mask > 0:
            next_pos = pos + bit_mask
            if next_pos <= self._n and self._tree[next_pos] < k:
                k -= self._tree[next_pos]
                pos = next_pos
            bit_mask >>= 1
        return pos  # 0-indexed
