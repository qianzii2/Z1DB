from __future__ import annotations
"""Robin Hood 开放寻址哈希表。"""
from typing import Any, Optional, Tuple
from metal.hash import z1hash64


class RobinHoodHashTable:
    __slots__ = ('_capacity', '_mask', '_keys', '_values',
                 '_dists', '_occupied', '_size', '_max_dist')

    def __init__(self, capacity: int = 64) -> None:
        cap = 16
        while cap < capacity: cap <<= 1
        self._capacity = cap; self._mask = cap - 1
        self._keys = [0] * cap
        self._values: list = [None] * cap
        self._dists = bytearray(cap)
        self._occupied = bytearray(cap)
        self._size = 0; self._max_dist = 0

    @property
    def size(self) -> int: return self._size
    @property
    def load_factor(self) -> float:
        return self._size / self._capacity if self._capacity else 0.0

    def get(self, key: int) -> Tuple[bool, Any]:
        h = self._hash(key)
        for dist in range(self._max_dist + 1):
            idx = (h + dist) & self._mask
            if not self._occupied[idx]: return (False, None)
            if self._dists[idx] < dist: return (False, None)
            if self._keys[idx] == key:
                return (True, self._values[idx])
        return (False, None)

    def put(self, key: int, value: Any) -> None:
        if self._size >= self._capacity * 85 // 100:
            self._grow()
        self._insert(key, value)

    def contains(self, key: int) -> bool:
        return self.get(key)[0]

    def remove(self, key: int) -> bool:
        h = self._hash(key)
        for dist in range(self._max_dist + 1):
            idx = (h + dist) & self._mask
            if not self._occupied[idx]: return False
            if self._dists[idx] < dist: return False
            if self._keys[idx] == key:
                self._occupied[idx] = 0
                self._size -= 1
                self._backshift(idx); return True
        return False

    def items(self):
        for i in range(self._capacity):
            if self._occupied[i]:
                yield (self._keys[i], self._values[i])

    def clear(self) -> None:
        for i in range(self._capacity):
            self._occupied[i] = 0
        self._size = 0; self._max_dist = 0

    def _insert(self, key, value):
        h = self._hash(key); dist = 0
        while True:
            idx = (h + dist) & self._mask
            if not self._occupied[idx]:
                self._keys[idx] = key
                self._values[idx] = value
                self._dists[idx] = dist
                self._occupied[idx] = 1
                self._size += 1
                if dist > self._max_dist:
                    self._max_dist = dist
                return
            if self._keys[idx] == key:
                self._values[idx] = value; return
            if self._dists[idx] < dist:
                key, self._keys[idx] = self._keys[idx], key
                value, self._values[idx] = (
                    self._values[idx], value)
                dist, self._dists[idx] = (
                    self._dists[idx], dist)
            dist += 1
            if dist > 128:
                self._grow()
                self._insert(key, value); return

    def _backshift(self, idx):
        while True:
            next_idx = (idx + 1) & self._mask
            if (not self._occupied[next_idx]
                    or self._dists[next_idx] == 0):
                break
            self._keys[idx] = self._keys[next_idx]
            self._values[idx] = self._values[next_idx]
            self._dists[idx] = self._dists[next_idx] - 1
            self._occupied[idx] = 1
            self._occupied[next_idx] = 0
            idx = next_idx

    def _hash(self, key: int) -> int:
        return z1hash64(
            key.to_bytes(8, 'little', signed=True)
        ) & self._mask

    def _grow(self):
        old_keys = self._keys; old_values = self._values
        old_occupied = self._occupied
        old_cap = self._capacity
        self._capacity = old_cap * 2
        self._mask = self._capacity - 1
        self._keys = [0] * self._capacity
        self._values = [None] * self._capacity
        self._dists = bytearray(self._capacity)
        self._occupied = bytearray(self._capacity)
        self._size = 0; self._max_dist = 0
        for i in range(old_cap):
            if old_occupied[i]:
                self._insert(old_keys[i], old_values[i])
