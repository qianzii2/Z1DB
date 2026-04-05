from __future__ import annotations
"""Count-Min Sketch — frequency estimation + APPROX_TOP_K.
Paper: Cormode & Muthukrishnan, 2005.
Space O(1/ε × log(1/δ)), update O(d), query O(d)."""
from typing import Any, List, Tuple
from metal.hash import murmur3_64


class CountMinSketch:
    """Frequency estimator. Never underestimates, may overestimate."""

    __slots__ = ('_width', '_depth', '_table', '_total')

    def __init__(self, width: int = 2048, depth: int = 5) -> None:
        self._width = width
        self._depth = depth
        self._table = [[0] * width for _ in range(depth)]
        self._total = 0

    def add(self, item: object, count: int = 1) -> None:
        """Increment count for item."""
        key = self._to_bytes(item)
        for d in range(self._depth):
            h = murmur3_64(key, seed=d * 0x9E3779B9) % self._width
            self._table[d][h] += count
        self._total += count

    def estimate(self, item: object) -> int:
        """Return estimated frequency (may overestimate, never underestimates)."""
        key = self._to_bytes(item)
        return min(
            self._table[d][murmur3_64(key, seed=d * 0x9E3779B9) % self._width]
            for d in range(self._depth))

    def top_k(self, candidates: list, k: int) -> List[Tuple[Any, int]]:
        """Return top-k items by estimated frequency from candidate list."""
        estimates = [(item, self.estimate(item)) for item in candidates]
        estimates.sort(key=lambda x: -x[1])
        return estimates[:k]

    def merge(self, other: CountMinSketch) -> None:
        """O(width × depth) merge."""
        if self._width != other._width or self._depth != other._depth:
            raise ValueError("incompatible sketches")
        for d in range(self._depth):
            for w in range(self._width):
                self._table[d][w] += other._table[d][w]
        self._total += other._total

    @property
    def total(self) -> int:
        return self._total

    @staticmethod
    def _to_bytes(item: object) -> bytes:
        if isinstance(item, bytes):
            return item
        if isinstance(item, str):
            return item.encode('utf-8')
        if isinstance(item, int):
            return item.to_bytes(8, 'little', signed=True)
        return str(item).encode('utf-8')
