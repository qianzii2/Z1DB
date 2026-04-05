from __future__ import annotations
"""Raw memory blocks — bypass Python object overhead with ctypes."""
import ctypes
import struct
from typing import List, Optional


class RawMemoryBlock:
    """Contiguous typed memory. int64 costs 8 bytes here vs 28 in Python."""

    _FMT = {'q': 'q', 'd': 'd', 'i': 'i', 'I': 'I', 'H': 'H', 'B': 'B', 'f': 'f'}

    __slots__ = ('_buf', '_fmt', '_code', '_itemsize', '_size', '_capacity')

    def __init__(self, dtype_code: str, capacity: int) -> None:
        self._code = dtype_code
        self._fmt = self._FMT.get(dtype_code, dtype_code)
        self._itemsize = struct.calcsize(self._fmt)
        self._capacity = capacity
        self._buf = ctypes.create_string_buffer(capacity * self._itemsize)
        self._size = 0

    @property
    def size(self) -> int:
        return self._size

    def append(self, value: object) -> None:
        if self._size >= self._capacity:
            self._grow()
        struct.pack_into(self._fmt, self._buf, self._size * self._itemsize, value)
        self._size += 1

    def batch_append(self, values: list) -> None:
        n = len(values)
        needed = self._size + n
        while needed > self._capacity:
            self._grow()
        fmt = f'{n}{self._code}'
        struct.pack_into(fmt, self._buf, self._size * self._itemsize, *values)
        self._size += n

    def get(self, index: int) -> object:
        if index < 0 or index >= self._size:
            raise IndexError(f"index {index} out of range [0, {self._size})")
        return struct.unpack_from(self._fmt, self._buf, index * self._itemsize)[0]

    def get_batch(self, start: int, count: int) -> list:
        if start + count > self._size:
            count = self._size - start
        if count <= 0:
            return []
        fmt = f'{count}{self._code}'
        return list(struct.unpack_from(fmt, self._buf, start * self._itemsize))

    def get_slice(self, start: int, end: int) -> memoryview:
        """Zero-copy slice — returns memoryview into raw buffer."""
        return memoryview(self._buf)[start * self._itemsize: end * self._itemsize]

    def set(self, index: int, value: object) -> None:
        struct.pack_into(self._fmt, self._buf, index * self._itemsize, value)

    def _grow(self) -> None:
        new_cap = max(self._capacity * 2, 16)
        new_buf = ctypes.create_string_buffer(new_cap * self._itemsize)
        ctypes.memmove(new_buf, self._buf, self._size * self._itemsize)
        self._buf = new_buf
        self._capacity = new_cap

    def raw_bytes(self) -> bytes:
        return bytes(self._buf[:self._size * self._itemsize])

    def __len__(self) -> int:
        return self._size
