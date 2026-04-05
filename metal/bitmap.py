from __future__ import annotations
"""动态增长位图 + 集合运算。优化popcount/to_indices性能。"""

from typing import List, Protocol, runtime_checkable


@runtime_checkable
class BitmapLike(Protocol):
    """位图协议。"""

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


# 逐字节popcount查表（0-255各有多少个1）
_BYTE_POPCOUNT = bytes(bin(i).count('1') for i in range(256))


class Bitmap:
    """动态增长位图，跟踪逻辑大小。"""

    __slots__ = ('_logical_size', '_data')

    def __init__(self, size: int = 0) -> None:
        self._logical_size = size
        self._data = bytearray((size + 7) // 8)

    @property
    def size(self) -> int:
        return self._logical_size

    def ensure_capacity(self, n: int) -> None:
        """扩容至少容纳n个位。"""
        if n > self._logical_size:
            needed = (n + 7) // 8
            if needed > len(self._data):
                self._data.extend(b'\x00' * (needed - len(self._data)))
            self._logical_size = n

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

    def copy(self) -> Bitmap:
        r = Bitmap(self._logical_size)
        r._data = bytearray(self._data)
        return r

    def append_from(self, other: Bitmap, other_length: int) -> None:
        """追加other的前other_length位到末尾。"""
        old = self._logical_size
        self.ensure_capacity(old + other_length)
        if old % 8 == 0:
            # 字节对齐 → 直接拷贝整字节
            full_bytes = other_length // 8
            start_byte = old // 8
            self._data[start_byte:start_byte + full_bytes] = \
                other._data[:full_bytes]
            # 尾部逐位
            for i in range(full_bytes * 8, other_length):
                if other.get_bit(i):
                    self.set_bit(old + i)
        else:
            # 非对齐回退逐位
            for i in range(other_length):
                if other.get_bit(i):
                    self.set_bit(old + i)

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

    def popcount(self) -> int:
        """统计置位数。查表法，比bin().count('1')快3-5x。"""
        full = self._logical_size // 8
        count = 0
        pc = _BYTE_POPCOUNT
        for i in range(full):
            count += pc[self._data[i]]
        tail = self._logical_size % 8
        if tail and full < len(self._data):
            count += pc[self._data[full] & ((1 << tail) - 1)]
        return count

    def to_indices(self) -> List[int]:
        """返回所有置位的索引。跳过全零字节加速。"""
        result: List[int] = []
        full = self._logical_size // 8
        for byte_idx in range(full):
            b = self._data[byte_idx]
            if b == 0:
                continue  # 快速跳过全零字节
            base = byte_idx << 3
            # 逐位提取
            while b:
                lowest = b & (-b)  # 最低置位
                bit_pos = lowest.bit_length() - 1
                result.append(base + bit_pos)
                b ^= lowest  # 清除最低置位
        # 尾部
        tail = self._logical_size % 8
        if tail and full < len(self._data):
            b = self._data[full] & ((1 << tail) - 1)
            base = full << 3
            while b:
                lowest = b & (-b)
                bit_pos = lowest.bit_length() - 1
                result.append(base + bit_pos)
                b ^= lowest
        return result

    @staticmethod
    def from_indices(indices: list[int], size: int) -> Bitmap:
        bm = Bitmap(size)
        for i in indices:
            bm.set_bit(i)
        return bm
