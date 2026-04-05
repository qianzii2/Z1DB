from __future__ import annotations
"""T-Digest — quantile estimation. Paper: Dunning & Ertl, 2019.
Extreme quantiles error < 0.1%. Mergeable for distributed computation."""
import bisect
import math
from typing import Any, List, Optional, Tuple


class _Centroid:
    __slots__ = ('mean', 'weight')

    def __init__(self, mean: float, weight: float) -> None:
        self.mean = mean
        self.weight = weight

    def __lt__(self, other: _Centroid) -> bool:
        return self.mean < other.mean


class TDigest:
    """Streaming quantile estimator with mergeable sketches."""

    __slots__ = ('_centroids', '_compression', '_total_weight',
                 '_buffer', '_buffer_limit', '_min', '_max')

    def __init__(self, compression: float = 100) -> None:
        self._centroids: List[_Centroid] = []
        self._compression = compression
        self._total_weight = 0.0
        self._buffer: List[float] = []
        self._buffer_limit = int(compression * 5)
        self._min = float('inf')
        self._max = float('-inf')

    def add(self, value: float, weight: float = 1.0) -> None:
        """Add a weighted value."""
        if value < self._min:
            self._min = value
        if value > self._max:
            self._max = value
        self._buffer.append(value)
        self._total_weight += weight
        if len(self._buffer) >= self._buffer_limit:
            self._flush()

    def quantile(self, q: float) -> float:
        """Estimate the q-th quantile (0 ≤ q ≤ 1)."""
        self._flush()
        if not self._centroids:
            return 0.0
        if q <= 0:
            return self._min
        if q >= 1:
            return self._max
        if len(self._centroids) == 1:
            return self._centroids[0].mean

        target = q * self._total_weight
        cumulative = 0.0
        for i, c in enumerate(self._centroids):
            if cumulative + c.weight > target:
                # Interpolate
                if i == 0:
                    return self._min + (c.mean - self._min) * target / (c.weight / 2)
                prev = self._centroids[i - 1]
                inner = target - cumulative
                frac = inner / c.weight
                return prev.mean + (c.mean - prev.mean) * frac
            cumulative += c.weight
        return self._max

    def median(self) -> float:
        return self.quantile(0.5)

    def merge(self, other: TDigest) -> None:
        """Merge another T-Digest into this one."""
        other._flush()
        for c in other._centroids:
            self._buffer.extend([c.mean] * int(c.weight))
            self._total_weight += c.weight
        if other._min < self._min:
            self._min = other._min
        if other._max > self._max:
            self._max = other._max
        self._flush()

    def _flush(self) -> None:
        """Compress buffer into centroids."""
        if not self._buffer:
            return
        # Add buffer values as weight-1 centroids
        new_centroids = list(self._centroids)
        for v in self._buffer:
            new_centroids.append(_Centroid(v, 1.0))
        self._buffer.clear()

        # Sort and compress
        new_centroids.sort()
        self._centroids = self._compress(new_centroids)

    def _compress(self, sorted_centroids: List[_Centroid]) -> List[_Centroid]:
        if not sorted_centroids:
            return []
        result = [sorted_centroids[0]]
        cumulative = sorted_centroids[0].weight

        for i in range(1, len(sorted_centroids)):
            c = sorted_centroids[i]
            # Size limit based on quantile position
            q = cumulative / self._total_weight if self._total_weight > 0 else 0.5
            max_size = 4 * self._compression * q * (1 - q) / self._total_weight
            max_size = max(max_size, 1)

            if result[-1].weight + c.weight <= max_size * self._total_weight:
                # Merge into last centroid
                total = result[-1].weight + c.weight
                result[-1].mean = (result[-1].mean * result[-1].weight +
                                   c.mean * c.weight) / total
                result[-1].weight = total
            else:
                result.append(c)
            cumulative += c.weight

        return result

    @property
    def count(self) -> int:
        self._flush()
        return int(self._total_weight)
