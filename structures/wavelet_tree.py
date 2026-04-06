from __future__ import annotations
"""小波树 — 区间分位数和区间频率查询 O(log σ)。
论文: Navarro, 2014 "Wavelet Trees for All"
用于窗口 PERCENTILE 免排序计算。σ = 字母表大小。"""
from typing import List, Optional, Tuple


class WaveletTree:
    """整数字母表 [0, sigma) 上的小波树。
    支持：
      quantile(l, r, k) — [l,r) 中第 k 小值 — O(log σ)
      rank(l, r, val)   — [l,r) 中 val 的出现次数 — O(log σ)
      range_freq(l, r, lo, hi) — [l,r) 中 [lo,hi] 范围的值计数 — O(log σ)
    """

    __slots__ = ('_n', '_sigma', '_root')

    def __init__(self, data: List[int],
                 sigma: Optional[int] = None) -> None:
        self._n = len(data)
        self._sigma = sigma if sigma is not None else (
            max(data) + 1 if data else 1)
        if self._n == 0:
            self._root = None
        else:
            self._root = self._build(data, 0, self._sigma)

    def quantile(self, l: int, r: int, k: int) -> int:
        """data[l:r] 中第 k 小值（0-indexed）。O(log σ)。"""
        if (self._root is None or l >= r
                or k < 0 or k >= r - l):
            return -1
        return self._quantile(
            self._root, l, r, k, 0, self._sigma)

    def rank(self, l: int, r: int, val: int) -> int:
        """data[l:r] 中 val 的出现次数。O(log σ)。"""
        if self._root is None or l >= r:
            return 0
        return self._rank(
            self._root, l, r, val, 0, self._sigma)

    def range_freq(self, l: int, r: int,
                   lo: int, hi: int) -> int:
        """data[l:r] 中值在 [lo, hi] 范围内的计数。O(log σ)。"""
        if self._root is None or l >= r or lo > hi:
            return 0
        return self._range_freq(
            self._root, l, r, lo, hi, 0, self._sigma)

    # ═══ 内部节点 ═══

    class _Node:
        __slots__ = ('bv', 'bv_rank', 'left', 'right', 'n')
        def __init__(self, n: int) -> None:
            self.n = n
            self.bv: List[int] = []       # 位向量：0=左子树，1=右子树
            self.bv_rank: List[int] = []   # bv 中 1 的前缀计数
            self.left: Optional[WaveletTree._Node] = None
            self.right: Optional[WaveletTree._Node] = None

    def _build(self, data, lo, hi):
        if lo >= hi - 1 or not data:
            node = self._Node(len(data))
            node.bv = []
            node.bv_rank = [0]
            return node
        mid = (lo + hi) // 2
        node = self._Node(len(data))
        left_data: List[int] = []
        right_data: List[int] = []
        node.bv_rank = [0]
        for val in data:
            if val < mid:
                node.bv.append(0)
                left_data.append(val)
            else:
                node.bv.append(1)
                right_data.append(val)
            node.bv_rank.append(
                node.bv_rank[-1] + node.bv[-1])
        if left_data:
            node.left = self._build(left_data, lo, mid)
        if right_data:
            node.right = self._build(right_data, mid, hi)
        return node

    def _rank1(self, node, pos):
        """bv[0:pos] 中 1 的数量。"""
        if pos <= 0:
            return 0
        if pos > len(node.bv_rank) - 1:
            pos = len(node.bv_rank) - 1
        return node.bv_rank[pos]

    def _rank0(self, node, pos):
        return pos - self._rank1(node, pos)

    def _quantile(self, node, l, r, k, lo, hi):
        if lo >= hi - 1:
            return lo
        mid = (lo + hi) // 2
        left_count = self._rank0(node, r) - self._rank0(node, l)
        if k < left_count:
            new_l = self._rank0(node, l)
            new_r = self._rank0(node, r)
            if node.left:
                return self._quantile(
                    node.left, new_l, new_r, k, lo, mid)
            return lo
        else:
            new_l = self._rank1(node, l)
            new_r = self._rank1(node, r)
            if node.right:
                return self._quantile(
                    node.right, new_l, new_r,
                    k - left_count, mid, hi)
            return mid

    def _rank(self, node, l, r, val, lo, hi):
        if lo >= hi - 1:
            return r - l
        mid = (lo + hi) // 2
        if val < mid:
            new_l = self._rank0(node, l)
            new_r = self._rank0(node, r)
            if node.left:
                return self._rank(
                    node.left, new_l, new_r, val, lo, mid)
            return 0
        else:
            new_l = self._rank1(node, l)
            new_r = self._rank1(node, r)
            if node.right:
                return self._rank(
                    node.right, new_l, new_r, val, mid, hi)
            return 0

    def _range_freq(self, node, l, r, qlo, qhi, lo, hi):
        if qlo <= lo and hi - 1 <= qhi:
            return r - l
        if lo >= hi - 1 or qlo >= hi or qhi < lo:
            return 0
        mid = (lo + hi) // 2
        count = 0
        if qlo < mid and node.left:
            new_l = self._rank0(node, l)
            new_r = self._rank0(node, r)
            count += self._range_freq(
                node.left, new_l, new_r, qlo, qhi, lo, mid)
        if qhi >= mid and node.right:
            new_l = self._rank1(node, l)
            new_r = self._rank1(node, r)
            count += self._range_freq(
                node.right, new_l, new_r, qlo, qhi, mid, hi)
        return count
