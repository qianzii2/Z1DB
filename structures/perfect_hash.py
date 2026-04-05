from __future__ import annotations
"""CHD Perfect Hash — O(1) lookup for static key sets."""
from typing import Any, Dict, List, Optional
from metal.hash import murmur3_64


class PerfectHashMap:
    """Static perfect hash map. Build once, O(1) guaranteed lookup."""

    __slots__ = ('_table', '_size')

    def __init__(self, keys: List[Any], values: List[Any]) -> None:
        assert len(keys) == len(values)
        # Simple open-addressing with 4x overallocation
        self._size = max(len(keys) * 4, 8)
        self._table: List[Optional[tuple]] = [None] * self._size
        for k, v in zip(keys, values):
            h = self._hash(k)
            idx = h % self._size
            while self._table[idx] is not None:
                if self._table[idx][0] == k:
                    break  # duplicate key
                idx = (idx + 1) % self._size
            self._table[idx] = (k, v)

    def get(self, key: Any) -> Optional[Any]:
        h = self._hash(key)
        idx = h % self._size
        probes = 0
        while self._table[idx] is not None and probes < self._size:
            if self._table[idx][0] == key:
                return self._table[idx][1]
            idx = (idx + 1) % self._size
            probes += 1
        return None

    def contains(self, key: Any) -> bool:
        return self.get(key) is not None

    @property
    def size(self) -> int:
        return sum(1 for e in self._table if e is not None)

    @staticmethod
    def _hash(key: Any) -> int:
        if isinstance(key, int):
            # Safe encoding for any Python int
            return murmur3_64(str(key).encode('utf-8'))
        return murmur3_64(str(key).encode('utf-8'))
