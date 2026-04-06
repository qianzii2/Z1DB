from __future__ import annotations
"""原始内存块 — 绕过 Python 对象开销。
Python int 占 28 字节，ctypes int64 仅占 8 字节。
大量数值存储时节省约 70% 内存。"""
import ctypes
import struct
from typing import List, Optional


class RawMemoryBlock:
    """连续类型化内存。TypedVector 在数据量 > 4096 时迁移到此。"""

    # 支持的格式码
    _FMT = {
        'q': 'q', 'd': 'd', 'i': 'i',
        'I': 'I', 'H': 'H', 'B': 'B', 'f': 'f',
    }

    __slots__ = ('_buf', '_fmt', '_code', '_itemsize',
                 '_size', '_capacity')

    def __init__(self, dtype_code: str,
                 capacity: int) -> None:
        self._code = dtype_code
        self._fmt = self._FMT.get(dtype_code, dtype_code)
        self._itemsize = struct.calcsize(self._fmt)
        self._capacity = capacity
        self._buf = ctypes.create_string_buffer(
            capacity * self._itemsize)
        self._size = 0

    @property
    def size(self) -> int:
        return self._size

    def append(self, value: object) -> None:
        """追加单个值。O(1) 均摊。"""
        if self._size >= self._capacity:
            self._grow()
        struct.pack_into(
            self._fmt, self._buf,
            self._size * self._itemsize, value)
        self._size += 1

    def batch_append(self, values: list) -> None:
        """批量追加。一次 pack_into 调用写入多个值。"""
        n = len(values)
        needed = self._size + n
        while needed > self._capacity:
            self._grow()
        fmt = f'{n}{self._code}'
        struct.pack_into(
            fmt, self._buf,
            self._size * self._itemsize, *values)
        self._size += n

    def get(self, index: int) -> object:
        """读取指定位置的值。O(1)。"""
        if index < 0 or index >= self._size:
            raise IndexError(
                f"索引 {index} 越界 [0, {self._size})")
        return struct.unpack_from(
            self._fmt, self._buf,
            index * self._itemsize)[0]

    def get_batch(self, start: int,
                  count: int) -> list:
        """批量读取 [start, start+count)。"""
        if start + count > self._size:
            count = self._size - start
        if count <= 0:
            return []
        fmt = f'{count}{self._code}'
        return list(struct.unpack_from(
            fmt, self._buf,
            start * self._itemsize))

    def get_slice(self, start: int,
                  end: int) -> memoryview:
        """零拷贝切片 — 返回 memoryview。"""
        return memoryview(self._buf)[
            start * self._itemsize:
            end * self._itemsize]

    def set(self, index: int,
            value: object) -> None:
        """修改指定位置的值。"""
        struct.pack_into(
            self._fmt, self._buf,
            index * self._itemsize, value)

    def _grow(self) -> None:
        """容量翻倍。"""
        new_cap = max(self._capacity * 2, 16)
        new_buf = ctypes.create_string_buffer(
            new_cap * self._itemsize)
        ctypes.memmove(
            new_buf, self._buf,
            self._size * self._itemsize)
        self._buf = new_buf
        self._capacity = new_cap

    def raw_bytes(self) -> bytes:
        """导出原始字节。"""
        return bytes(
            self._buf[:self._size * self._itemsize])

    def __len__(self) -> int:
        return self._size

    def __repr__(self) -> str:
        return (f"RawMemoryBlock(code='{self._code}', "
                f"size={self._size}, cap={self._capacity})")
