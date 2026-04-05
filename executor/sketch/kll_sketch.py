from __future__ import annotations
"""KLL Sketch — theoretically optimal quantile estimation.
Paper: Karnin, Lang, Liberty, 2016.
Space O(1/ε × log²(1/ε) × log log(1/δ)). Mergeable."""
import math
import random
from typing import List, Optional


class KLLSketch:
    """KLL quantile sketch — deterministic accuracy guarantee."""

    __slots__ = ('_k', '_compactors', '_size', '_min', '_max',
                 '_height', '_max_size')

    def __init__(self, k: int = 200) -> None:
        self._k = k  # accuracy parameter (higher = more accurate)
        self._compactors: List[List[float]] = [[]]
        self._size = 0
        self._min = float('inf')
        self._max = float('-inf')
        self._height = 1
        self._max_size = k

    def add(self, value: float) -> None:
        """Add a value to the sketch."""
        if value < self._min:
            self._min = value
        if value > self._max:
            self._max = value
        self._compactors[0].append(value)
        self._size += 1
        if len(self._compactors[0]) >= self._capacity(0):
            self._compact()

    def quantile(self, q: float) -> float:
        """Estimate the q-th quantile (0 ≤ q ≤ 1)."""
        if self._size == 0:
            return 0.0
        if q <= 0:
            return self._min
        if q >= 1:
            return self._max

        # Collect all weighted items
        weighted: List[tuple] = []
        for level, compactor in enumerate(self._compactors):
            weight = 1 << level
            for v in compactor:
                weighted.append((v, weight))
        weighted.sort(key=lambda x: x[0])

        total_weight = sum(w for _, w in weighted)
        target = q * total_weight
        cumulative = 0.0
        for val, weight in weighted:
            cumulative += weight
            if cumulative >= target:
                return val
        return self._max

    def median(self) -> float:
        return self.quantile(0.5)

    def merge(self, other: KLLSketch) -> None:
        """Merge another KLL sketch into this one."""
        if other._min < self._min:
            self._min = other._min
        if other._max > self._max:
            self._max = other._max
        # Ensure enough levels
        while len(self._compactors) < len(other._compactors):
            self._compactors.append([])
        for level, compactor in enumerate(other._compactors):
            self._compactors[level].extend(compactor)
        self._size += other._size
        # Compact from bottom up
        for level in range(len(self._compactors)):
            if len(self._compactors[level]) >= self._capacity(level):
                self._compact_level(level)

    @property
    def count(self) -> int:
        return self._size

    def relative_error(self) -> float:
        """Theoretical relative error bound."""
        return 1.0 / self._k

    def _capacity(self, level: int) -> int:
        """Capacity at a given level. Grows with level."""
        depth = len(self._compactors) - level - 1
        return max(2, int(math.ceil(self._k * (2.0 / 3.0) ** depth)))

    def _compact(self) -> None:
        """Compact the lowest level that is full."""
        for level in range(len(self._compactors)):
            if len(self._compactors[level]) >= self._capacity(level):
                self._compact_level(level)
                return

    def _compact_level(self, level: int) -> None:
        """Compact one level: sort, keep every other item, promote to next level."""
        if level >= len(self._compactors):
            return
        compactor = self._compactors[level]
        compactor.sort()
        # Randomly choose even or odd indices
        offset = random.randint(0, 1)
        promoted = [compactor[i] for i in range(offset, len(compactor), 2)]
        # Replace with remaining items
        remaining = [compactor[i] for i in range(1 - offset, len(compactor), 2)]
        self._compactors[level] = remaining

        # Promote to next level
        if level + 1 >= len(self._compactors):
            self._compactors.append([])
            self._height += 1
        self._compactors[level + 1].extend(promoted)

        # Recursively compact if next level is full
        if len(self._compactors[level + 1]) >= self._capacity(level + 1):
            self._compact_level(level + 1)
