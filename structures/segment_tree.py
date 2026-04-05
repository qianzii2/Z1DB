from __future__ import annotations
"""Segment Tree — range aggregate O(log n), point update O(log n).
Used for window function SUM/AVG/COUNT with fixed frames."""
from typing import Any, Callable, List, Optional
import operator


class SegmentTree:
    """Generic segment tree for any associative aggregate."""

    __slots__ = ('_n', '_tree', '_op', '_identity')

    def __init__(self, data: list, op: Callable = operator.add,
                 identity: Any = 0) -> None:
        self._n = len(data)
        self._op = op
        self._identity = identity
        self._tree = [identity] * (4 * max(self._n, 1))
        if self._n > 0:
            self._build(data, 1, 0, self._n - 1)

    def _build(self, data: list, node: int, start: int, end: int) -> None:
        if start == end:
            self._tree[node] = data[start]
            return
        mid = (start + end) // 2
        self._build(data, 2 * node, start, mid)
        self._build(data, 2 * node + 1, mid + 1, end)
        self._tree[node] = self._op(self._tree[2 * node], self._tree[2 * node + 1])

    def query(self, l: int, r: int) -> Any:
        """Aggregate over [l, r] inclusive. O(log n)."""
        if l > r or l >= self._n or r < 0:
            return self._identity
        l = max(l, 0)
        r = min(r, self._n - 1)
        return self._query(1, 0, self._n - 1, l, r)

    def _query(self, node: int, start: int, end: int, l: int, r: int) -> Any:
        if r < start or end < l:
            return self._identity
        if l <= start and end <= r:
            return self._tree[node]
        mid = (start + end) // 2
        left = self._query(2 * node, start, mid, l, r)
        right = self._query(2 * node + 1, mid + 1, end, l, r)
        return self._op(left, right)

    def update(self, pos: int, value: Any) -> None:
        """Point update. O(log n)."""
        if 0 <= pos < self._n:
            self._update(1, 0, self._n - 1, pos, value)

    def _update(self, node: int, start: int, end: int,
                pos: int, value: Any) -> None:
        if start == end:
            self._tree[node] = value
            return
        mid = (start + end) // 2
        if pos <= mid:
            self._update(2 * node, start, mid, pos, value)
        else:
            self._update(2 * node + 1, mid + 1, end, pos, value)
        self._tree[node] = self._op(self._tree[2 * node], self._tree[2 * node + 1])


class MinSegmentTree(SegmentTree):
    """Segment tree specialized for MIN queries."""
    def __init__(self, data: list) -> None:
        super().__init__(data, op=min, identity=float('inf'))


class MaxSegmentTree(SegmentTree):
    """Segment tree specialized for MAX queries."""
    def __init__(self, data: list) -> None:
        super().__init__(data, op=max, identity=float('-inf'))
