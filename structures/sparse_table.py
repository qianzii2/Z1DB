from __future__ import annotations
"""稀疏表 — O(n log n) 构建，O(1) 区间 MIN/MAX 查询。不可变。
用于窗口函数的固定帧 MIN/MAX（帧大小固定时比线段树快）。"""
import math
from typing import Any, Callable, List


class SparseTable:
    """不可变区间查询。仅适用于幂等运算（min/max）。"""

    __slots__ = ('_n', '_k', '_table', '_log', '_op')

    def __init__(self, data: list,
                 op: Callable = min) -> None:
        self._op = op
        self._n = len(data)
        if self._n == 0:
            self._k = 0; self._table = []; self._log = []
            return
        self._k = max(1, int(math.log2(self._n)) + 1)
        # 预计算 log2 查表
        self._log = [0] * (self._n + 1)
        for i in range(2, self._n + 1):
            self._log[i] = self._log[i // 2] + 1
        # 构建稀疏表：table[j][i] = op(data[i..i+2^j-1])
        self._table: List[list] = [list(data)]
        for j in range(1, self._k):
            prev = self._table[j - 1]
            row = []
            half = 1 << (j - 1)
            for i in range(self._n - (1 << j) + 1):
                row.append(self._op(prev[i], prev[i + half]))
            self._table.append(row)

    def query(self, l: int, r: int) -> Any:
        """O(1) 区间查询 [l, r]。仅对幂等运算有效。"""
        if l > r or self._n == 0:
            return None
        l = max(l, 0)
        r = min(r, self._n - 1)
        length = r - l + 1
        k = self._log[length]
        return self._op(
            self._table[k][l],
            self._table[k][r - (1 << k) + 1])


class SparseTableMin(SparseTable):
    def __init__(self, data: list) -> None:
        super().__init__(data, op=min)


class SparseTableMax(SparseTable):
    def __init__(self, data: list) -> None:
        super().__init__(data, op=max)
