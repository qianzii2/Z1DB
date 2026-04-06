from __future__ import annotations
"""高级哈希 — Zobrist 增量哈希 / CuckooHashMap / WriteCombiningBuffer。"""
import random
from typing import Any, Dict, List, Optional, Tuple
from metal.hash import z1hash64


# ═══ Zobrist 增量哈希 ═══

class ZobristHasher:
    """增量哈希：更新单列值时 O(1) 重算行哈希。"""

    def __init__(self, num_columns: int,
                 seed: int = 42) -> None:
        self._num_cols = num_columns
        self._rng = random.Random(seed)
        self._salt: List[int] = [
            self._rng.getrandbits(64)
            for _ in range(num_columns)]

    def hash_value(self, col: int, value: Any) -> int:
        if value is None:
            return 0
        return z1hash64(
            str(value).encode('utf-8'),
            seed=self._salt[col])

    def hash_row(self, values: list) -> int:
        h = 0
        for col, val in enumerate(values):
            h ^= self.hash_value(col, val)
        return h

    def update_hash(self, old_hash: int, col: int,
                    old_value: Any, new_value: Any) -> int:
        return (old_hash
                ^ self.hash_value(col, old_value)
                ^ self.hash_value(col, new_value))


# ═══ Cuckoo Hash Map ═══

class CuckooHashMap:
    """int64 key → value 的 O(1) 确定性查找哈希表。"""
    MAX_KICKS = 500

    def __init__(self, capacity: int = 256) -> None:
        self._cap = max(16, capacity)
        self._table1: List[Optional[Tuple[int, Any]]] = [None] * self._cap
        self._table2: List[Optional[Tuple[int, Any]]] = [None] * self._cap
        self._size = 0
        self._seed1 = 0
        self._seed2 = 0x9E3779B97F4A7C15

    def get(self, key: int) -> Optional[Any]:
        h1 = self._hash1(key)
        entry = self._table1[h1]
        if entry is not None and entry[0] == key:
            return entry[1]
        h2 = self._hash2(key)
        entry = self._table2[h2]
        if entry is not None and entry[0] == key:
            return entry[1]
        return None

    def contains(self, key: int) -> bool:
        return self.get(key) is not None

    def put(self, key: int, value: Any) -> None:
        h1 = self._hash1(key)
        if self._table1[h1] is not None and self._table1[h1][0] == key:
            self._table1[h1] = (key, value)
            return
        h2 = self._hash2(key)
        if self._table2[h2] is not None and self._table2[h2][0] == key:
            self._table2[h2] = (key, value)
            return
        entry = (key, value)
        for _ in range(self.MAX_KICKS):
            h1 = self._hash1(entry[0])
            if self._table1[h1] is None:
                self._table1[h1] = entry
                self._size += 1
                return
            entry, self._table1[h1] = self._table1[h1], entry
            h2 = self._hash2(entry[0])
            if self._table2[h2] is None:
                self._table2[h2] = entry
                self._size += 1
                return
            entry, self._table2[h2] = self._table2[h2], entry
        self._rehash()
        self.put(entry[0], entry[1])

    def remove(self, key: int) -> bool:
        h1 = self._hash1(key)
        if self._table1[h1] is not None and self._table1[h1][0] == key:
            self._table1[h1] = None
            self._size -= 1
            return True
        h2 = self._hash2(key)
        if self._table2[h2] is not None and self._table2[h2][0] == key:
            self._table2[h2] = None
            self._size -= 1
            return True
        return False

    @property
    def size(self) -> int:
        return self._size

    def _hash1(self, key: int) -> int:
        return z1hash64(
            key.to_bytes(8, 'little', signed=True),
            seed=self._seed1) % self._cap

    def _hash2(self, key: int) -> int:
        return z1hash64(
            key.to_bytes(8, 'little', signed=True),
            seed=self._seed2) % self._cap

    def _rehash(self) -> None:
        old1, old2 = self._table1, self._table2
        self._cap *= 2
        self._table1 = [None] * self._cap
        self._table2 = [None] * self._cap
        self._seed1 = random.getrandbits(64)
        self._seed2 = random.getrandbits(64)
        self._size = 0
        for entry in old1:
            if entry is not None:
                self.put(entry[0], entry[1])
        for entry in old2:
            if entry is not None:
                self.put(entry[0], entry[1])


# ═══ WriteCombiningBuffer ═══

class WriteCombiningBuffer:
    """分区写入缓冲，减少缓存行竞争。用于 Radix JOIN 分区阶段。"""
    BUFFER_SIZE = 8

    def __init__(self, num_partitions: int) -> None:
        self._num_parts = num_partitions
        self._buffers: List[List[Any]] = [[] for _ in range(num_partitions)]
        self._outputs: List[List[Any]] = [[] for _ in range(num_partitions)]

    def write(self, partition: int, value: Any) -> None:
        buf = self._buffers[partition]
        buf.append(value)
        if len(buf) >= self.BUFFER_SIZE:
            self._flush(partition)

    def _flush(self, partition: int) -> None:
        self._outputs[partition].extend(self._buffers[partition])
        self._buffers[partition] = []

    def flush_all(self) -> None:
        for i in range(self._num_parts):
            if self._buffers[i]:
                self._flush(i)

    def get_partition(self, partition: int) -> List[Any]:
        self.flush_all()
        return self._outputs[partition]

    @property
    def partitions(self) -> int:
        return self._num_parts
