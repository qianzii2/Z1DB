from __future__ import annotations
"""Bloom Filter — probabilistic set membership. Mergeable.
False positive rate ≈ (1 - e^(-kn/m))^k. Supports merge via OR."""
import math
from metal.hash import murmur3_128


class BloomFilter:
    """Space-efficient probabilistic filter. No false negatives."""

    __slots__ = ('_m', '_k', '_bits', '_count')

    def __init__(self, expected_items: int = 1000,
                 fp_rate: float = 0.01) -> None:
        if expected_items < 1:
            expected_items = 1
        self._m = max(64, int(-expected_items * math.log(fp_rate) / (math.log(2) ** 2)))
        self._k = max(1, int(self._m / expected_items * math.log(2)))
        self._bits = bytearray((self._m + 7) // 8)
        self._count = 0

    def add(self, item: object) -> None:
        """Add an item. item can be bytes, str, or int."""
        raw = self._to_bytes(item)
        h1, h2 = murmur3_128(raw)
        for i in range(self._k):
            pos = (h1 + i * h2) % self._m
            self._bits[pos >> 3] |= (1 << (pos & 7))
        self._count += 1

    def contains(self, item: object) -> bool:
        """Check membership. May return false positive, never false negative."""
        raw = self._to_bytes(item)
        h1, h2 = murmur3_128(raw)
        for i in range(self._k):
            pos = (h1 + i * h2) % self._m
            if not (self._bits[pos >> 3] & (1 << (pos & 7))):
                return False
        return True

    def merge(self, other: BloomFilter) -> None:
        """O(m) merge — for parallel/distributed scenarios."""
        if self._m != other._m:
            raise ValueError("cannot merge filters of different sizes")
        for i in range(len(self._bits)):
            self._bits[i] |= other._bits[i]
        self._count += other._count

    @property
    def count(self) -> int:
        return self._count

    @property
    def estimated_fp_rate(self) -> float:
        if self._count == 0:
            return 0.0
        return (1 - math.exp(-self._k * self._count / self._m)) ** self._k

    def size_bytes(self) -> int:
        return len(self._bits)

    @staticmethod
    def _to_bytes(item: object) -> bytes:
        if isinstance(item, bytes):
            return item
        if isinstance(item, str):
            return item.encode('utf-8')
        if isinstance(item, int):
            return item.to_bytes(8, 'little', signed=True)
        if isinstance(item, float):
            import struct
            return struct.pack('d', item)
        return str(item).encode('utf-8')
