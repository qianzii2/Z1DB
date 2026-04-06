from __future__ import annotations
"""Roaring Bitmap — 自适应压缩位图。
论文: Chambi et al., 2016
稀疏区域用 uint16 有序数组，稠密区域用 8KB 位图。
AND/OR 操作比 bytearray 快 10-100x（大稀疏集合）。
容器切换阈值：4096 个元素。"""
import bisect
from typing import List, Optional


class _ArrayContainer:
    """稀疏容器 — uint16 有序数组。基数 < 4096 时使用。"""
    __slots__ = ('_data',)

    def __init__(self) -> None:
        self._data: list[int] = []

    def add(self, value: int) -> None:
        i = bisect.bisect_left(self._data, value)
        if i < len(self._data) and self._data[i] == value:
            return
        self._data.insert(i, value)

    def remove(self, value: int) -> bool:
        i = bisect.bisect_left(self._data, value)
        if i < len(self._data) and self._data[i] == value:
            self._data.pop(i)
            return True
        return False

    def contains(self, value: int) -> bool:
        i = bisect.bisect_left(self._data, value)
        return i < len(self._data) and self._data[i] == value

    @property
    def cardinality(self) -> int:
        return len(self._data)

    def to_list(self) -> list[int]:
        return list(self._data)

    def and_op(self, other: '_ArrayContainer') -> '_ArrayContainer':
        r = _ArrayContainer()
        i = j = 0
        while i < len(self._data) and j < len(other._data):
            if self._data[i] == other._data[j]:
                r._data.append(self._data[i]); i += 1; j += 1
            elif self._data[i] < other._data[j]:
                i += 1
            else:
                j += 1
        return r

    def or_op(self, other: '_ArrayContainer') -> '_ArrayContainer':
        r = _ArrayContainer()
        i = j = 0
        while i < len(self._data) and j < len(other._data):
            if self._data[i] == other._data[j]:
                r._data.append(self._data[i]); i += 1; j += 1
            elif self._data[i] < other._data[j]:
                r._data.append(self._data[i]); i += 1
            else:
                r._data.append(other._data[j]); j += 1
        r._data.extend(self._data[i:])
        r._data.extend(other._data[j:])
        return r

    def should_convert(self) -> bool:
        """基数 >= 4096 时应切换为位图容器。"""
        return len(self._data) >= 4096


class _BitmapContainer:
    """稠密容器 — 8KB 位图（65536 位）。"""
    __slots__ = ('_bits',)

    def __init__(self) -> None:
        self._bits = bytearray(8192)

    def add(self, value: int) -> None:
        self._bits[value >> 3] |= (1 << (value & 7))

    def remove(self, value: int) -> bool:
        if self._bits[value >> 3] & (1 << (value & 7)):
            self._bits[value >> 3] &= ~(1 << (value & 7))
            return True
        return False

    def contains(self, value: int) -> bool:
        return bool(self._bits[value >> 3] & (1 << (value & 7)))

    @property
    def cardinality(self) -> int:
        return sum(bin(b).count('1') for b in self._bits)

    def to_list(self) -> list[int]:
        result = []
        for byte_idx in range(8192):
            b = self._bits[byte_idx]
            if b:
                base = byte_idx * 8
                for bit in range(8):
                    if b & (1 << bit):
                        result.append(base + bit)
        return result

    def and_op(self, other: '_BitmapContainer') -> '_BitmapContainer':
        r = _BitmapContainer()
        for i in range(8192):
            r._bits[i] = self._bits[i] & other._bits[i]
        return r

    def or_op(self, other: '_BitmapContainer') -> '_BitmapContainer':
        r = _BitmapContainer()
        for i in range(8192):
            r._bits[i] = self._bits[i] | other._bits[i]
        return r

    def should_convert(self) -> bool:
        """基数 < 4096 时应切回数组容器。"""
        return self.cardinality < 4096


_Container = _ArrayContainer | _BitmapContainer


def _to_bitmap(ac: _ArrayContainer) -> _BitmapContainer:
    bc = _BitmapContainer()
    for v in ac._data:
        bc.add(v)
    return bc


def _to_array(bc: _BitmapContainer) -> _ArrayContainer:
    ac = _ArrayContainer()
    ac._data = bc.to_list()
    return ac


def _op_containers(a: _Container, b: _Container,
                   op: str) -> _Container:
    """组合两个容器，自动转换类型。"""
    if isinstance(a, _ArrayContainer) and isinstance(b, _ArrayContainer):
        r = a.and_op(b) if op == 'AND' else a.or_op(b)
        if r.should_convert():
            return _to_bitmap(r)
        return r
    ba = a if isinstance(a, _BitmapContainer) else _to_bitmap(a)
    bb = b if isinstance(b, _BitmapContainer) else _to_bitmap(b)
    r = ba.and_op(bb) if op == 'AND' else ba.or_op(bb)
    if r.should_convert():
        return _to_array(r)
    return r


class RoaringBitmap:
    """Roaring Bitmap — 自适应压缩位图。
    按高 16 位分区，每个分区用 Array 或 Bitmap 容器。"""

    __slots__ = ('_containers',)

    def __init__(self) -> None:
        self._containers: dict[int, _Container] = {}

    def add(self, value: int) -> None:
        high = value >> 16
        low = value & 0xFFFF
        if high not in self._containers:
            self._containers[high] = _ArrayContainer()
        c = self._containers[high]
        c.add(low)
        if isinstance(c, _ArrayContainer) and c.should_convert():
            self._containers[high] = _to_bitmap(c)

    def remove(self, value: int) -> bool:
        high = value >> 16
        if high not in self._containers:
            return False
        c = self._containers[high]
        result = c.remove(value & 0xFFFF)
        if isinstance(c, _BitmapContainer) and c.should_convert():
            self._containers[high] = _to_array(c)
        if c.cardinality == 0:
            del self._containers[high]
        return result

    def contains(self, value: int) -> bool:
        high = value >> 16
        if high not in self._containers:
            return False
        return self._containers[high].contains(value & 0xFFFF)

    def add_range(self, start: int, end: int) -> None:
        """添加 [start, end) 范围所有值。"""
        for v in range(start, end):
            self.add(v)

    def cardinality(self) -> int:
        return sum(c.cardinality for c in self._containers.values())

    def to_indices(self) -> list[int]:
        result = []
        for high in sorted(self._containers.keys()):
            base = high << 16
            result.extend(
                base + v for v in self._containers[high].to_list())
        return result

    def and_op(self, other: 'RoaringBitmap') -> 'RoaringBitmap':
        r = RoaringBitmap()
        for high in self._containers:
            if high in other._containers:
                c = _op_containers(
                    self._containers[high],
                    other._containers[high], 'AND')
                if c.cardinality > 0:
                    r._containers[high] = c
        return r

    def or_op(self, other: 'RoaringBitmap') -> 'RoaringBitmap':
        r = RoaringBitmap()
        for high in self._containers:
            if high in other._containers:
                r._containers[high] = _op_containers(
                    self._containers[high],
                    other._containers[high], 'OR')
            else:
                r._containers[high] = self._containers[high]
        for high in other._containers:
            if high not in self._containers:
                r._containers[high] = other._containers[high]
        return r

    def not_op(self, universe: int) -> 'RoaringBitmap':
        """取补集（相对于 [0, universe)）。"""
        r = RoaringBitmap()
        for v in range(universe):
            if not self.contains(v):
                r.add(v)
        return r

    def is_empty(self) -> bool:
        return len(self._containers) == 0
