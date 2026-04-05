from __future__ import annotations
"""MemTable — in-memory sorted buffer for LSM-Tree writes.
Paper: O'Neil et al., 1996 "The Log-Structured Merge-Tree"

Writes go to MemTable first (zero random I/O).
When full (64K rows), flush to sorted SSTable on disk."""
import bisect
from typing import Any, Dict, List, Optional, Tuple

_DEFAULT_CAPACITY = 65536


class MemTable:
    """Sorted in-memory write buffer. Insert O(log n), scan O(n)."""

    __slots__ = ('_entries', '_size', '_capacity', '_frozen')

    def __init__(self, capacity: int = _DEFAULT_CAPACITY) -> None:
        self._entries: List[Tuple[Any, list]] = []  # sorted by key
        self._size = 0
        self._capacity = capacity
        self._frozen = False

    @property
    def size(self) -> int:
        return self._size

    @property
    def is_full(self) -> bool:
        return self._size >= self._capacity

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    def freeze(self) -> None:
        """Mark as immutable. No more writes allowed."""
        self._frozen = True

    def put(self, key: Any, row: list) -> None:
        """Insert or update a row. O(log n) via bisect."""
        if self._frozen:
            raise RuntimeError("MemTable is frozen")
        idx = bisect.bisect_left(self._entries, (key,))
        if idx < len(self._entries) and self._entries[idx][0] == key:
            self._entries[idx] = (key, list(row))
        else:
            self._entries.insert(idx, (key, list(row)))
            self._size += 1

    def get(self, key: Any) -> Optional[list]:
        """Point lookup. O(log n)."""
        idx = bisect.bisect_left(self._entries, (key,))
        if idx < len(self._entries) and self._entries[idx][0] == key:
            return self._entries[idx][1]
        return None

    def delete(self, key: Any) -> bool:
        """Mark key as deleted (tombstone). O(log n)."""
        idx = bisect.bisect_left(self._entries, (key,))
        if idx < len(self._entries) and self._entries[idx][0] == key:
            self._entries[idx] = (key, None)  # None = tombstone
            return True
        # Insert tombstone even if key doesn't exist in this memtable
        self._entries.insert(idx, (key, None))
        self._size += 1
        return False

    def scan(self) -> List[Tuple[Any, Optional[list]]]:
        """Return all entries sorted by key. O(n)."""
        return list(self._entries)

    def scan_range(self, lo: Any, hi: Any) -> List[Tuple[Any, Optional[list]]]:
        """Range scan [lo, hi]. O(log n + k)."""
        start = bisect.bisect_left(self._entries, (lo,))
        result = []
        for i in range(start, len(self._entries)):
            k, v = self._entries[i]
            if k > hi:
                break
            result.append((k, v))
        return result

    def clear(self) -> None:
        self._entries.clear()
        self._size = 0
        self._frozen = False
