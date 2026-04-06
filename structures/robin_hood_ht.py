from __future__ import annotations
"""Robin Hood 开放寻址哈希表。
核心思想：插入时如果当前元素的探测距离大于已有元素，则"窃取"位置。
效果：最大探测距离 O(log n)，方差极小。
用于 HashAgg 的整数 key 分组。"""
from typing import Any, Optional, Tuple
from metal.hash import z1hash64


class RobinHoodHashTable:
    """int64 key → Any value 的开放寻址哈希表。
    负载因子 85% 自动扩容。探测距离超过 128 强制扩容。"""

    __slots__ = ('_capacity', '_mask', '_keys', '_values',
                 '_dists', '_occupied', '_size', '_max_dist')

    def __init__(self, capacity: int = 64) -> None:
        cap = 16
        while cap < capacity:
            cap <<= 1
        self._capacity = cap
        self._mask = cap - 1
        self._keys = [0] * cap
        self._values: list = [None] * cap
        self._dists = bytearray(cap)       # 每个槽的探测距离
        self._occupied = bytearray(cap)    # 是否占用
        self._size = 0
        self._max_dist = 0

    @property
    def size(self) -> int:
        return self._size

    @property
    def load_factor(self) -> float:
        return self._size / self._capacity if self._capacity else 0.0

    def get(self, key: int) -> Tuple[bool, Any]:
        """O(1) 均摊查找。返回 (found, value)。"""
        h = self._hash(key)
        for dist in range(self._max_dist + 1):
            idx = (h + dist) & self._mask
            if not self._occupied[idx]:
                return (False, None)
            if self._dists[idx] < dist:
                return (False, None)  # 不可能在更远处
            if self._keys[idx] == key:
                return (True, self._values[idx])
        return (False, None)

    def put(self, key: int, value: Any) -> None:
        """插入或更新。负载过高时自动扩容。"""
        if self._size >= self._capacity * 85 // 100:
            self._grow()
        self._insert(key, value)

    def contains(self, key: int) -> bool:
        return self.get(key)[0]

    def remove(self, key: int) -> bool:
        """删除 + 后移回填（保持探测距离不变性）。"""
        h = self._hash(key)
        for dist in range(self._max_dist + 1):
            idx = (h + dist) & self._mask
            if not self._occupied[idx]:
                return False
            if self._dists[idx] < dist:
                return False
            if self._keys[idx] == key:
                self._occupied[idx] = 0
                self._size -= 1
                self._backshift(idx)
                return True
        return False

    def items(self):
        """遍历所有 (key, value) 对。"""
        for i in range(self._capacity):
            if self._occupied[i]:
                yield (self._keys[i], self._values[i])

    def clear(self) -> None:
        for i in range(self._capacity):
            self._occupied[i] = 0
        self._size = 0
        self._max_dist = 0

    def _insert(self, key, value):
        """Robin Hood 插入：探测距离大的窃取探测距离小的位置。"""
        h = self._hash(key)
        dist = 0
        while True:
            idx = (h + dist) & self._mask
            if not self._occupied[idx]:
                # 空槽：直接插入
                self._keys[idx] = key
                self._values[idx] = value
                self._dists[idx] = dist
                self._occupied[idx] = 1
                self._size += 1
                if dist > self._max_dist:
                    self._max_dist = dist
                return
            if self._keys[idx] == key:
                # 已存在：更新值
                self._values[idx] = value
                return
            if self._dists[idx] < dist:
                # 窃取：当前元素探测距离更大，夺取此位置
                key, self._keys[idx] = self._keys[idx], key
                value, self._values[idx] = (
                    self._values[idx], value)
                dist, self._dists[idx] = (
                    self._dists[idx], dist)
            dist += 1
            if dist > 128:
                self._grow()
                self._insert(key, value)
                return

    def _backshift(self, idx):
        """删除后回填：将后续元素前移以保持探测距离。"""
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
        """扩容并重新哈希。"""
        old_keys = self._keys
        old_values = self._values
        old_occupied = self._occupied
        old_cap = self._capacity
        self._capacity = old_cap * 2
        self._mask = self._capacity - 1
        self._keys = [0] * self._capacity
        self._values = [None] * self._capacity
        self._dists = bytearray(self._capacity)
        self._occupied = bytearray(self._capacity)
        self._size = 0
        self._max_dist = 0
        for i in range(old_cap):
            if old_occupied[i]:
                self._insert(old_keys[i], old_values[i])
