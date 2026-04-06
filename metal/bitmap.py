from __future__ import annotations
"""动态位图 — 64 位字优化 + 池化 + 线程安全。
[安全] recycle 时复制 _data，切断外部引用。
[线程] _POOL 用 threading.Lock 保护。"""
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

_WORD_OPT_THRESHOLD = 64

# 线程安全的 Bitmap 池
_POOL_LOCK = threading.Lock()
_POOL: List[bytearray] = []  # 存 bytearray 而非 Bitmap（切断引用）
_POOL_MAX = 512


@runtime_checkable
class BitmapLike(Protocol):
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


_BYTE_POPCOUNT = bytes(bin(i).count('1') for i in range(256))


class Bitmap:
    __slots__ = ('_logical_size', '_data')

    def __init__(self, size: int = 0) -> None:
        self._logical_size = size
        self._data = bytearray((size + 7) // 8)

    @staticmethod
    def pooled(size: int) -> 'Bitmap':
        """从池中取 bytearray 构建 Bitmap。池中存的是裸 bytearray，
        不是 Bitmap 对象，因此即使旧 Bitmap 仍被引用也不会被污染。"""
        byte_needed = (size + 7) // 8
        with _POOL_LOCK:
            for i, buf in enumerate(_POOL):
                if len(buf) >= byte_needed:
                    _POOL.pop(i)
                    # 清零复用的 bytearray
                    for j in range(byte_needed):
                        buf[j] = 0
                    bm = Bitmap.__new__(Bitmap)
                    bm._logical_size = size
                    bm._data = buf[:byte_needed] if len(buf) > byte_needed * 2 else buf
                    # 截断多余字节（如果差距不大就保留，避免频繁分配）
                    if len(bm._data) > byte_needed:
                        # 多余字节清零即可，不截断（避免 bytearray 重新分配）
                        for j in range(byte_needed, len(bm._data)):
                            bm._data[j] = 0
                    bm._logical_size = size
                    return bm
        return Bitmap(size)

    @staticmethod
    def recycle(bm: 'Bitmap') -> None:
        """归还 bytearray 到池中。
        关键：池中存的是 _data 的**副本**，原 Bitmap 对象的 _data 被置空。
        这样即使外部仍持有旧 Bitmap 引用，其 _data 已不可用（get_bit 返回 False）。"""
        with _POOL_LOCK:
            if len(_POOL) < _POOL_MAX:
                # 存 _data 的副本到池中
                _POOL.append(bytearray(bm._data))
        # 将原 Bitmap 标记为不可用
        bm._data = bytearray(0)
        bm._logical_size = 0

    @property
    def size(self) -> int:
        return self._logical_size

    def ensure_capacity(self, n: int) -> None:
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

    def copy(self) -> 'Bitmap':
        r = Bitmap(self._logical_size)
        r._data = bytearray(self._data)
        return r

    def append_from(self, other: 'Bitmap', other_length: int) -> None:
        old = self._logical_size
        self.ensure_capacity(old + other_length)
        if old % 8 == 0:
            full_bytes = other_length // 8
            start_byte = old // 8
            self._data[start_byte:start_byte + full_bytes] = other._data[:full_bytes]
            for i in range(full_bytes * 8, other_length):
                if other.get_bit(i):
                    self.set_bit(old + i)
        else:
            for i in range(other_length):
                if other.get_bit(i):
                    self.set_bit(old + i)

    def and_op(self, other: 'Bitmap') -> 'Bitmap':
        size = min(self._logical_size, other._logical_size)
        r = Bitmap.pooled(size)
        bc = (size + 7) // 8
        i = 0
        if bc >= _WORD_OPT_THRESHOLD:
            while i + 8 <= bc:
                wa = int.from_bytes(self._data[i:i+8], 'little')
                wb = int.from_bytes(other._data[i:i+8], 'little')
                r._data[i:i+8] = (wa & wb).to_bytes(8, 'little')
                i += 8
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
                wa = int.from_bytes(self._data[i:i+8], 'little')
                wb = int.from_bytes(other._data[i:i+8], 'little')
                r._data[i:i+8] = (wa | wb).to_bytes(8, 'little')
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
        tail = self._logical_size % 8
        if tail and r._data:
            r._data[-1] &= (1 << tail) - 1
        return r

    def popcount(self) -> int:
        """统计置位数。64 位字批量处理 + 尾部字节查表。"""
        total = 0
        data = self._data
        byte_count = self._logical_size // 8
        tail_bits = self._logical_size % 8
        pc = _BYTE_POPCOUNT

        # 64 位字批量路径
        i = 0
        while i + 8 <= byte_count:
            word = int.from_bytes(data[i:i + 8], 'little')
            # SWAR popcount (Hamming weight)
            word = word - ((word >> 1) & 0x5555555555555555)
            word = (word & 0x3333333333333333) + ((word >> 2) & 0x3333333333333333)
            word = (word + (word >> 4)) & 0x0F0F0F0F0F0F0F0F
            total += ((word * 0x0101010101010101) >> 56) & 0xFF
            i += 8

        # 剩余字节逐字节查表
        while i < byte_count:
            total += pc[data[i]]
            i += 1

        # 尾部不足一字节
        if tail_bits and byte_count < len(data):
            total += pc[data[byte_count] & ((1 << tail_bits) - 1)]

        return total

    def to_indices(self) -> List[int]:
        result = []
        full = self._logical_size // 8
        for byte_idx in range(full):
            b = self._data[byte_idx]
            if b == 0:
                continue
            base = byte_idx << 3
            while b:
                lowest = b & (-b)
                result.append(base + lowest.bit_length() - 1)
                b ^= lowest
        tail = self._logical_size % 8
        if tail and full < len(self._data):
            b = self._data[full] & ((1 << tail) - 1)
            base = full << 3
            while b:
                lowest = b & (-b)
                result.append(base + lowest.bit_length() - 1)
                b ^= lowest
        return result

    def select(self, k: int) -> int:
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

    def gather_values(self, data_list: list) -> list:
        if not _HAS_BITMAGIC or len(data_list) < 8:
            return [data_list[i] for i in self.to_indices()]
        result = []
        full = min(self._logical_size, len(data_list))
        for byte_idx in range(full // 8):
            b = self._data[byte_idx]
            if b == 0:
                continue
            base = byte_idx * 8
            result.extend(_bitmap_gather(data_list[base:base + 8], b))
        for i in range(full // 8 * 8, full):
            if self.get_bit(i):
                result.append(data_list[i])
        return result

    def gather_with_nulls(self, data_list: list,
                          null_bitmap: 'Bitmap') -> tuple:
        """一次遍历提取值 + 构建新 null bitmap。"""
        indices = self.to_indices()
        n = len(indices)
        new_nulls = Bitmap.pooled(n)
        values = []
        for j, i in enumerate(indices):
            values.append(data_list[i])
            if null_bitmap.get_bit(i):
                new_nulls.set_bit(j)
        return values, new_nulls

    def to_roaring(self) -> Optional[object]:
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
    def from_indices(indices: 'list[int]', size: int) -> 'Bitmap':
        bm = Bitmap(size)
        for i in indices:
            bm.set_bit(i)
        return bm
