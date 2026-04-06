from __future__ import annotations
"""树状数组 (Fenwick Tree / BIT) — 前缀和 O(log n)，单点更新 O(log n)。
常数因子比线段树小。用于窗口函数的前缀累加 SUM。"""
from typing import List


class FenwickTree:
    """二叉索引树，支持前缀和与第 k 大查询。"""

    __slots__ = ('_n', '_tree')

    def __init__(self, n: int) -> None:
        self._n = n
        self._tree = [0] * (n + 1)

    @staticmethod
    def from_list(data: list) -> 'FenwickTree':
        """从列表 O(n) 构建。"""
        ft = FenwickTree(len(data))
        for i, v in enumerate(data):
            ft._tree[i + 1] = v
        for i in range(1, len(data) + 1):
            j = i + (i & (-i))
            if j <= len(data):
                ft._tree[j] += ft._tree[i]
        return ft

    def update(self, i: int, delta: int) -> None:
        """位置 i 加 delta。O(log n)。"""
        i += 1
        while i <= self._n:
            self._tree[i] += delta
            i += i & (-i)

    def prefix_sum(self, i: int) -> int:
        """[0, i] 的和。O(log n)。"""
        s = 0
        i += 1
        while i > 0:
            s += self._tree[i]
            i -= i & (-i)
        return s

    def range_sum(self, l: int, r: int) -> int:
        """[l, r] 的和。O(log n)。"""
        if l > r:
            return 0
        s = self.prefix_sum(r)
        if l > 0:
            s -= self.prefix_sum(l - 1)
        return s

    def find_kth(self, k: int) -> int:
        """找最小的 i 使得 prefix_sum(i) >= k。O(log n)。
        要求所有值非负。"""
        pos = 0
        bit_mask = 1
        while bit_mask <= self._n:
            bit_mask <<= 1
        bit_mask >>= 1
        while bit_mask > 0:
            next_pos = pos + bit_mask
            if (next_pos <= self._n
                    and self._tree[next_pos] < k):
                k -= self._tree[next_pos]
                pos = next_pos
            bit_mask >>= 1
        return pos
