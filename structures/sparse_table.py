from __future__ import annotations
"""Sparse Table — O(n log n) build, O(1) range MIN/MAX query. Immutable.
Used for window MIN/MAX with fixed frames."""
import math
from typing import Any, Callable, List


class SparseTable:
    """Immutable range MIN/MAX in O(1) after O(n log n) preprocessing."""

    __slots__ = ('_n', '_k', '_table', '_log', '_op')

    def __init__(self, data: list, op: Callable = min) -> None:
        self._op = op
        self._n = len(data)
        if self._n == 0:
            self._k = 0
            self._table = []
            self._log = []
            return
        self._k = max(1, int(math.log2(self._n)) + 1)
        self._log = [0] * (self._n + 1)
        for i in range(2, self._n + 1):
            self._log[i] = self._log[i // 2] + 1
        self._table: List[list] = [list(data)]
        for j in range(1, self._k):
            prev = self._table[j - 1]
            row = []
            span = 1 << j
            half = 1 << (j - 1)
            for i in range(self._n - span + 1):
                row.append(self._op(prev[i], prev[i + half]))
            self._table.append(row)

    def query(self, l: int, r: int) -> Any:
        """O(1) query for [l, r] inclusive. Only valid for idempotent ops (min, max)."""
        if l > r or self._n == 0:
            return None
        l = max(l, 0)
        r = min(r, self._n - 1)
        length = r - l + 1
        k = self._log[length]
        return self._op(self._table[k][l], self._table[k][r - (1 << k) + 1])


class SparseTableMin(SparseTable):
    def __init__(self, data: list) -> None:
        super().__init__(data, op=min)


class SparseTableMax(SparseTable):
    def __init__(self, data: list) -> None:
        super().__init__(data, op=max)
