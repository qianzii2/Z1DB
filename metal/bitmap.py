from __future__ import annotations
"""Dynamic-growth bitmap with set-theoretic operations."""

from typing import List, Protocol, runtime_checkable


@runtime_checkable
class BitmapLike(Protocol):
    """Structural protocol for bitmap-like objects."""

    def set_bit(self, i: int) -> None: ...
    def clear_bit(self, i: int) -> None: ...
    def get_bit(self, i: int) -> bool: ...
    def and_op(self, other: BitmapLike) -> BitmapLike: ...
    def or_op(self, other: BitmapLike) -> BitmapLike: ...
    def not_op(self) -> BitmapLike: ...
    def popcount(self) -> int: ...
    def to_indices(self) -> List[int]: ...

    @property
    def size(self) -> int: ...


class Bitmap:
    """Dynamically growable bitmap that tracks its logical size."""

    __slots__ = ('_logical_size', '_data')

    def __init__(self, size: int = 0) -> None:
        self._logical_size = size
        self._data = bytearray((size + 7) // 8)

    # -- properties ----------------------------------------------------
    @property
    def size(self) -> int:
        return self._logical_size

    # -- capacity ------------------------------------------------------
    def ensure_capacity(self, n: int) -> None:
        """Grow so that at least *n* bits are addressable."""
        if n > self._logical_size:
            needed = (n + 7) // 8
            if needed > len(self._data):
                self._data.extend(b'\x00' * (needed - len(self._data)))
            self._logical_size = n

    # -- single-bit ops ------------------------------------------------
    def set_bit(self, i: int) -> None:
        if i >= self._logical_size:
            self.ensure_capacity(i + 1)
        self._data[i >> 3] |= (1 << (i & 7))

    def clear_bit(self, i: int) -> None:
        if i < len(self._data) * 8:
            self._data[i >> 3] &= ~(1 << (i & 7))

    def get_bit(self, i: int) -> bool:
        if i >= self._logical_size:
            return False
        return bool(self._data[i >> 3] & (1 << (i & 7)))

    # -- bulk ----------------------------------------------------------
    def copy(self) -> Bitmap:
        r = Bitmap(self._logical_size)
        r._data = bytearray(self._data)
        return r

    def append_from(self, other: Bitmap, other_length: int) -> None:
        """Append the first *other_length* bits of *other* to the tail."""
        old = self._logical_size
        self.ensure_capacity(old + other_length)
        for i in range(other_length):
            if other.get_bit(i):
                self.set_bit(old + i)

    # -- set-theoretic -------------------------------------------------
    def and_op(self, other: Bitmap) -> Bitmap:
        size = min(self._logical_size, other._logical_size)
        r = Bitmap(size)
        byte_count = (size + 7) // 8
        for i in range(byte_count):
            a = self._data[i] if i < len(self._data) else 0
            b = other._data[i] if i < len(other._data) else 0
            r._data[i] = a & b
        return r

    def or_op(self, other: Bitmap) -> Bitmap:
        size = max(self._logical_size, other._logical_size)
        r = Bitmap(size)
        byte_count = (size + 7) // 8
        for i in range(byte_count):
            a = self._data[i] if i < len(self._data) else 0
            b = other._data[i] if i < len(other._data) else 0
            r._data[i] = a | b
        return r

    def not_op(self) -> Bitmap:
        r = Bitmap(self._logical_size)
        for i in range(len(self._data)):
            r._data[i] = (~self._data[i]) & 0xFF
        tail = self._logical_size % 8
        if tail and r._data:
            r._data[-1] &= (1 << tail) - 1
        return r

    # -- aggregates ----------------------------------------------------
    def popcount(self) -> int:
        count = 0
        full = self._logical_size // 8
        for i in range(full):
            count += bin(self._data[i]).count('1')
        tail = self._logical_size % 8
        if tail and full < len(self._data):
            count += bin(self._data[full] & ((1 << tail) - 1)).count('1')
        return count

    def to_indices(self) -> List[int]:
        return [i for i in range(self._logical_size) if self.get_bit(i)]

    # -- factory -------------------------------------------------------
    @staticmethod
    def from_indices(indices: list[int], size: int) -> Bitmap:
        bm = Bitmap(size)
        for i in indices:
            bm.set_bit(i)
        return bm
