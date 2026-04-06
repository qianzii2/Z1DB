from __future__ import annotations
"""Bloom Filter — 概率集合判断。
[P08] 新增 to_bytes/from_bytes 序列化支持。"""
import math
from metal.hash import z1hash128


class BloomFilter:
    __slots__ = ('_m', '_k', '_bits', '_count')

    def __init__(self, expected_items: int = 1000,
                 fp_rate: float = 0.01) -> None:
        if expected_items < 1:
            expected_items = 1
        self._m = max(64, int(
            -expected_items * math.log(fp_rate) / (math.log(2) ** 2)))
        self._k = max(1, int(self._m / expected_items * math.log(2)))
        self._bits = bytearray((self._m + 7) // 8)
        self._count = 0

    def add(self, item: object) -> None:
        key = self._to_bytes(item)
        h1, h2 = z1hash128(key)
        for i in range(self._k):
            pos = (h1 + i * h2) % self._m
            self._bits[pos >> 3] |= (1 << (pos & 7))
        self._count += 1

    def contains(self, item: object) -> bool:
        key = self._to_bytes(item)
        h1, h2 = z1hash128(key)
        for i in range(self._k):
            pos = (h1 + i * h2) % self._m
            if not (self._bits[pos >> 3] & (1 << (pos & 7))):
                return False
        return True

    def merge(self, other: 'BloomFilter') -> None:
        if self._m != other._m:
            raise ValueError("无法合并不同大小的 Bloom Filter")
        for i in range(len(self._bits)):
            self._bits[i] |= other._bits[i]
        self._count += other._count

    # [P08] 序列化/反序列化
    def to_bytes(self) -> bytes:
        """序列化为 bytes：[m:4B][k:4B][count:4B][bits...]"""
        import struct
        header = struct.pack('<III', self._m, self._k, self._count)
        return header + bytes(self._bits)

    @staticmethod
    def from_bytes(data: bytes) -> 'BloomFilter':
        """从 bytes 反序列化。"""
        import struct
        if len(data) < 12:
            return BloomFilter(1)
        m, k, count = struct.unpack_from('<III', data, 0)
        bf = BloomFilter.__new__(BloomFilter)
        bf._m = m
        bf._k = k
        bf._count = count
        bf._bits = bytearray(data[12:])
        # 确保 bits 长度正确
        expected = (m + 7) // 8
        if len(bf._bits) < expected:
            bf._bits.extend(b'\x00' * (expected - len(bf._bits)))
        return bf

    @property
    def count(self) -> int:
        return self._count

    @property
    def estimated_fp_rate(self) -> float:
        if self._count == 0:
            return 0.0
        return (1 - math.exp(-self._k * self._count / self._m)) ** self._k

    def size_bytes(self) -> int:
        return len(self._bits)

    @staticmethod
    def _to_bytes(item: object) -> bytes:
        if isinstance(item, bytes):
            return item
        if isinstance(item, str):
            return item.encode('utf-8')
        if isinstance(item, int):
            return item.to_bytes(8, 'little', signed=True)
        if isinstance(item, float):
            import struct
            return struct.pack('d', item)
        return str(item).encode('utf-8')
