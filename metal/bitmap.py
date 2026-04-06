from __future__ import annotations
"""动态位图 — 64 位字优化 popcount + 池化。
[D05] 池化重设计：池中存 bytearray 引用，recycle 后原 Bitmap 不可用。
[P10] popcount 使用 SWAR 在 Python int 上操作（CPython 中已是最优解）。"""
import threading
from typing import List, Optional, Protocol, runtime_checkable

try:
    from metal.bitmagic import (
        select64 as _select64, pdep as _pdep,
        pext as _pext, bitmap_gather as _bitmap_gather)
    _HAS_BITMAGIC = True
except ImportError:
    _HAS_BITMAGIC = False

try:
    from structures.roaring_bitmap import RoaringBitmap
    _HAS_ROARING = True
except ImportError:
    _HAS_ROARING = False

# 64 位字优化阈值（字节数）
_WORD_OPT_THRESHOLD = 64

# 线程安全的池
_POOL_LOCK = threading.Lock()
_POOL: List[bytearray] = []
_POOL_MAX = 512


@runtime_checkable
class BitmapLike(Protocol):
    """位图统一接口（Bitmap 和 RoaringBitmap 共同遵守）。"""
    def set_bit(self, i: int) -> None: ...
    def clear_bit(self, i: int) -> None: ...
    def get_bit(self, i: int) -> bool: ...
    def and_op(self, other: 'BitmapLike') -> 'BitmapLike': ...
    def or_op(self, other: 'BitmapLike') -> 'BitmapLike': ...
    def not_op(self) -> 'BitmapLike': ...
    def popcount(self) -> int: ...
    def to_indices(self) -> List[int]: ...
    @property
    def size(self) -> int: ...


# 字节 popcount 查表（256 个字节值 → 置位数）
_BYTE_POPCOUNT = bytes(bin(i).count('1') for i in range(256))


class Bitmap:
    """动态位图。支持池化复用、SWAR popcount、Roaring 转换。"""

    __slots__ = ('_logical_size', '_data')

    def __init__(self, size: int = 0) -> None:
        self._logical_size = size
        self._data = bytearray((size + 7) // 8)

    # ═══ 池化 [D05] ═══

    @staticmethod
    def pooled(size: int) -> 'Bitmap':
        """从池中取 bytearray 构建 Bitmap。取出后清零复用。
        池中存的是裸 bytearray，不是 Bitmap 对象。"""
        byte_needed = (size + 7) // 8
        with _POOL_LOCK:
            for i, buf in enumerate(_POOL):
                if len(buf) >= byte_needed:
                    _POOL.pop(i)
                    # 清零复用
                    for j in range(byte_needed):
                        buf[j] = 0
                    bm = Bitmap.__new__(Bitmap)
                    bm._logical_size = size
                    # 如果池中 buf 远大于需要，截断避免浪费
                    if len(buf) > byte_needed * 2:
                        bm._data = bytearray(buf[:byte_needed])
                    else:
                        bm._data = buf
                        # 多余字节清零
                        for j in range(byte_needed, len(bm._data)):
                            bm._data[j] = 0
                    return bm
        return Bitmap(size)

    @staticmethod
    def recycle(bm: 'Bitmap') -> None:
        """归还 Bitmap 的底层 bytearray 到池中。
        重要：recycle 后原 Bitmap 立即不可用（_data 被清空）。"""
        with _POOL_LOCK:
            if len(_POOL) < _POOL_MAX and len(bm._data) > 0:
                _POOL.append(bm._data)
        # 使原 Bitmap 不可用
        bm._data = bytearray(0)
        bm._logical_size = 0

    # ═══ 基本操作 ═══

    @property
    def size(self) -> int:
        return self._logical_size

    def ensure_capacity(self, n: int) -> None:
        """确保能存储至少 n 位。"""
        if n > self._logical_size:
            needed = (n + 7) // 8
            if needed > len(self._data):
                self._data.extend(
                    b'\x00' * (needed - len(self._data)))
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

    def copy(self) -> 'Bitmap':
        r = Bitmap(self._logical_size)
        r._data = bytearray(self._data)
        return r

    def append_from(self, other: 'Bitmap',
                    other_length: int) -> None:
        """将 other 的前 other_length 位追加到尾部。"""
        old = self._logical_size
        self.ensure_capacity(old + other_length)
        if old % 8 == 0:
            # 字节对齐：可以直接复制整字节
            full_bytes = other_length // 8
            start_byte = old // 8
            self._data[start_byte:start_byte + full_bytes] = \
                other._data[:full_bytes]
            # 剩余不足一字节的位逐个复制
            for i in range(full_bytes * 8, other_length):
                if other.get_bit(i):
                    self.set_bit(old + i)
        else:
            for i in range(other_length):
                if other.get_bit(i):
                    self.set_bit(old + i)

    # ═══ 集合运算（64位字批量优化）═══

    def and_op(self, other: 'Bitmap') -> 'Bitmap':
        size = min(self._logical_size, other._logical_size)
        r = Bitmap.pooled(size)
        bc = (size + 7) // 8
        i = 0
        # 64 位字批量路径
        if bc >= _WORD_OPT_THRESHOLD:
            while i + 8 <= bc:
                wa = int.from_bytes(
                    self._data[i:i + 8], 'little')
                wb = int.from_bytes(
                    other._data[i:i + 8], 'little')
                r._data[i:i + 8] = (wa & wb).to_bytes(
                    8, 'little')
                i += 8
        # 逐字节尾部
        while i < bc:
            a = self._data[i] if i < len(self._data) else 0
            b = other._data[i] if i < len(other._data) else 0
            r._data[i] = a & b
            i += 1
        return r

    def or_op(self, other: 'Bitmap') -> 'Bitmap':
        size = max(self._logical_size, other._logical_size)
        r = Bitmap.pooled(size)
        bc = (size + 7) // 8
        al, bl = len(self._data), len(other._data)
        i = 0
        ml = min(al, bl)
        if ml >= _WORD_OPT_THRESHOLD:
            while i + 8 <= ml:
                wa = int.from_bytes(
                    self._data[i:i + 8], 'little')
                wb = int.from_bytes(
                    other._data[i:i + 8], 'little')
                r._data[i:i + 8] = (wa | wb).to_bytes(
                    8, 'little')
                i += 8
        while i < bc:
            a = self._data[i] if i < al else 0
            b = other._data[i] if i < bl else 0
            r._data[i] = a | b
            i += 1
        return r

    def not_op(self) -> 'Bitmap':
        r = Bitmap.pooled(self._logical_size)
        for i in range(len(self._data)):
            r._data[i] = (~self._data[i]) & 0xFF
        # 清除尾部多余位
        tail = self._logical_size % 8
        if tail and r._data:
            r._data[-1] &= (1 << tail) - 1
        return r

    # ═══ 统计 [P10] ═══

    def popcount(self) -> int:
        """统计置位数。64 位字 SWAR + 字节查表混合。
        SWAR 在 Python int 上操作已是 CPython 中的最优解
        （比 bin().count('1') 快约 3x，因为避免了字符串分配）。"""
        total = 0
        data = self._data
        byte_count = self._logical_size // 8
        tail_bits = self._logical_size % 8
        pc = _BYTE_POPCOUNT

        # 64 位字批量 SWAR popcount
        i = 0
        while i + 8 <= byte_count:
            word = int.from_bytes(data[i:i + 8], 'little')
            # SWAR Hamming weight
            word -= ((word >> 1) & 0x5555555555555555)
            word = ((word & 0x3333333333333333) +
                    ((word >> 2) & 0x3333333333333333))
            word = ((word + (word >> 4)) &
                    0x0F0F0F0F0F0F0F0F)
            total += (
                (word * 0x0101010101010101) >> 56) & 0xFF
            i += 8

        # 剩余字节逐字节查表
        while i < byte_count:
            total += pc[data[i]]
            i += 1

        # 尾部不足一字节
        if tail_bits and byte_count < len(data):
            total += pc[
                data[byte_count] & ((1 << tail_bits) - 1)]

        return total

    def to_indices(self) -> List[int]:
        """返回所有置位位的索引列表。"""
        result = []
        full = self._logical_size // 8
        for byte_idx in range(full):
            b = self._data[byte_idx]
            if b == 0:
                continue
            base = byte_idx << 3
            while b:
                lowest = b & (-b)
                result.append(
                    base + lowest.bit_length() - 1)
                b ^= lowest
        # 尾部
        tail = self._logical_size % 8
        if tail and full < len(self._data):
            b = self._data[full] & ((1 << tail) - 1)
            base = full << 3
            while b:
                lowest = b & (-b)
                result.append(
                    base + lowest.bit_length() - 1)
                b ^= lowest
        return result

    def select(self, k: int) -> int:
        """找第 k 个置位位的位置（0-indexed）。无则返回 -1。"""
        remaining = k
        for byte_idx in range(len(self._data)):
            b = self._data[byte_idx]
            cnt = _BYTE_POPCOUNT[b]
            if remaining < cnt:
                for bit_pos in range(8):
                    if b & (1 << bit_pos):
                        if remaining == 0:
                            return (byte_idx << 3) + bit_pos
                        remaining -= 1
            remaining -= cnt
        return -1

    # ═══ 批量提取 ═══

    def gather_values(self, data_list: list) -> list:
        """按位图提取 data_list 中对应位为 1 的元素。"""
        if not _HAS_BITMAGIC or len(data_list) < 8:
            return [data_list[i] for i in self.to_indices()]
        result = []
        full = min(self._logical_size, len(data_list))
        for byte_idx in range(full // 8):
            b = self._data[byte_idx]
            if b == 0:
                continue
            base = byte_idx * 8
            result.extend(_bitmap_gather(
                data_list[base:base + 8], b))
        for i in range(full // 8 * 8, full):
            if self.get_bit(i):
                result.append(data_list[i])
        return result

    def gather_with_nulls(self, data_list: list,
                          null_bitmap: 'Bitmap') -> tuple:
        """一次遍历提取值 + 构建新 null bitmap。
        返回 (filtered_values, new_null_bitmap)。"""
        indices = self.to_indices()
        n = len(indices)
        new_nulls = Bitmap.pooled(n)
        values = []
        for j, i in enumerate(indices):
            values.append(data_list[i])
            if null_bitmap.get_bit(i):
                new_nulls.set_bit(j)
        return values, new_nulls

    # ═══ Roaring 转换 ═══

    def to_roaring(self) -> Optional[object]:
        """转为 RoaringBitmap（大稀疏位图时内存更优）。"""
        if not _HAS_ROARING:
            return None
        rb = RoaringBitmap()
        for byte_idx in range(len(self._data)):
            b = self._data[byte_idx]
            if b == 0:
                continue
            base = byte_idx << 3
            while b:
                lowest = b & (-b)
                rb.add(base + lowest.bit_length() - 1)
                b ^= lowest
        return rb

    @staticmethod
    def from_roaring(rb: object, size: int) -> 'Bitmap':
        bm = Bitmap(size)
        if hasattr(rb, 'to_indices'):
            for i in rb.to_indices():
                if i < size:
                    bm.set_bit(i)
        return bm

    @staticmethod
    def from_indices(indices: 'list[int]',
                     size: int) -> 'Bitmap':
        """从索引列表构建位图。"""
        bm = Bitmap(size)
        for i in indices:
            bm.set_bit(i)
        return bm
