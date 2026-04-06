from __future__ import annotations
"""MemTable — LSM-Tree 的内存写缓冲。
论文: O'Neil et al., 1996 "The Log-Structured Merge-Tree"
写入先进 MemTable（零随机 I/O）。满时刷盘为有序 SSTable。"""
import bisect
from typing import Any, List, Optional, Tuple

_DEFAULT_CAPACITY = 65536


class MemTable:
    """有序内存写缓冲。插入 O(log n)，扫描 O(n)。
    row=None 表示 tombstone（已删除标记）。"""

    __slots__ = ('_entries', '_size', '_capacity', '_frozen')

    def __init__(self, capacity: int = _DEFAULT_CAPACITY
                 ) -> None:
        self._entries: List[Tuple[Any, list]] = []
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
        """标记为不可变。后续不允许写入。"""
        self._frozen = True

    def put(self, key: Any, row: list) -> None:
        """插入或更新。row=None 为 tombstone。O(log n)。"""
        if self._frozen:
            raise RuntimeError("MemTable 已冻结")
        idx = bisect.bisect_left(self._entries, (key,))
        if (idx < len(self._entries)
                and self._entries[idx][0] == key):
            self._entries[idx] = (key, list(row) if row is not None else None)
        else:
            self._entries.insert(
                idx, (key, list(row) if row is not None else None))
            self._size += 1

    def get(self, key: Any) -> Optional[list]:
        """点查。O(log n)。"""
        idx = bisect.bisect_left(self._entries, (key,))
        if (idx < len(self._entries)
                and self._entries[idx][0] == key):
            return self._entries[idx][1]
        return None

    def delete(self, key: Any) -> bool:
        """标记删除（写入 tombstone）。O(log n)。"""
        idx = bisect.bisect_left(self._entries, (key,))
        if (idx < len(self._entries)
                and self._entries[idx][0] == key):
            self._entries[idx] = (key, None)
            return True
        self._entries.insert(idx, (key, None))
        self._size += 1
        return False

    def scan(self) -> List[Tuple[Any, Optional[list]]]:
        """按 key 顺序返回所有条目。O(n)。"""
        return list(self._entries)

    def scan_range(self, lo: Any,
                   hi: Any) -> List[Tuple[Any, Optional[list]]]:
        """范围扫描 [lo, hi]。O(log n + k)。"""
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
