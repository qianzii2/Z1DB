from __future__ import annotations
"""Count-Min Sketch — 频率估计。"""
from typing import Any, List, Tuple
from metal.hash import z1hash64


class CountMinSketch:
    __slots__ = ('_width', '_depth', '_table', '_total')

    def __init__(self, width: int = 2048,
                 depth: int = 5) -> None:
        self._width = width; self._depth = depth
        self._table = [[0] * width for _ in range(depth)]
        self._total = 0

    def add(self, item: object, count: int = 1) -> None:
        key = self._to_bytes(item)
        for d in range(self._depth):
            h = z1hash64(
                key, seed=d * 0x9E3779B9) % self._width
            self._table[d][h] += count
        self._total += count

    def estimate(self, item: object) -> int:
        key = self._to_bytes(item)
        return min(
            self._table[d][z1hash64(
                key, seed=d * 0x9E3779B9) % self._width]
            for d in range(self._depth))

    def top_k(self, candidates: list,
              k: int) -> List[Tuple[Any, int]]:
        estimates = [(item, self.estimate(item))
                     for item in candidates]
        estimates.sort(key=lambda x: -x[1])
        return estimates[:k]

    def merge(self, other: CountMinSketch) -> None:
        if (self._width != other._width
                or self._depth != other._depth):
            raise ValueError("incompatible sketches")
        for d in range(self._depth):
            for w in range(self._width):
                self._table[d][w] += other._table[d][w]
        self._total += other._total

    @property
    def total(self) -> int: return self._total

    @staticmethod
    def _to_bytes(item: object) -> bytes:
        if isinstance(item, bytes): return item
        if isinstance(item, str):
            return item.encode('utf-8')
        if isinstance(item, int):
            return item.to_bytes(8, 'little', signed=True)
        return str(item).encode('utf-8')
