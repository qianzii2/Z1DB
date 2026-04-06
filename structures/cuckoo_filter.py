from __future__ import annotations
"""Cuckoo Filter — 支持删除的概率过滤器。
相比 Bloom Filter 优势：支持 delete 操作，空间效率更高。
用于索引管理器的行删除追踪。"""
import random
from metal.hash import z1hash64


class CuckooFilter:
    """Cuckoo Filter：每桶 4 个指纹槽，最多 500 次踢出。"""

    BUCKET_SIZE = 4
    MAX_KICKS = 500
    __slots__ = ('_num_buckets', '_table', '_count')

    def __init__(self, capacity: int = 1024) -> None:
        self._num_buckets = max(4, capacity // self.BUCKET_SIZE)
        # 对齐到 2 的幂次（位运算取模更快）
        n = 1
        while n < self._num_buckets:
            n <<= 1
        self._num_buckets = n
        self._table = [[0] * self.BUCKET_SIZE
                       for _ in range(self._num_buckets)]
        self._count = 0

    def add(self, item: bytes) -> bool:
        """添加元素。返回 False 表示容量已满。"""
        if isinstance(item, int):
            item = item.to_bytes(8, 'little', signed=True)
        elif isinstance(item, str):
            item = item.encode('utf-8')
        fp = self._fingerprint(item)
        if fp == 0:
            fp = 1  # 指纹 0 保留为空槽标记
        i1 = z1hash64(item) % self._num_buckets
        i2 = (i1 ^ z1hash64(bytes([fp]))) % self._num_buckets
        # 尝试直接插入
        for idx in (i1, i2):
            for slot in range(self.BUCKET_SIZE):
                if self._table[idx][slot] == 0:
                    self._table[idx][slot] = fp
                    self._count += 1
                    return True
        # 两个桶都满 → 踢出
        idx = random.choice([i1, i2])
        for _ in range(self.MAX_KICKS):
            slot = random.randint(0, self.BUCKET_SIZE - 1)
            fp, self._table[idx][slot] = (
                self._table[idx][slot], fp)
            idx = (idx ^ z1hash64(bytes([fp]))) % self._num_buckets
            for s in range(self.BUCKET_SIZE):
                if self._table[idx][s] == 0:
                    self._table[idx][s] = fp
                    self._count += 1
                    return True
        return False  # 容量已满

    def contains(self, item: bytes) -> bool:
        """查询。False = 一定不在，True = 可能在。"""
        if isinstance(item, int):
            item = item.to_bytes(8, 'little', signed=True)
        elif isinstance(item, str):
            item = item.encode('utf-8')
        fp = self._fingerprint(item)
        if fp == 0:
            fp = 1
        i1 = z1hash64(item) % self._num_buckets
        i2 = (i1 ^ z1hash64(bytes([fp]))) % self._num_buckets
        for idx in (i1, i2):
            if fp in self._table[idx]:
                return True
        return False

    def delete(self, item: bytes) -> bool:
        """删除元素。返回是否找到并删除。"""
        if isinstance(item, int):
            item = item.to_bytes(8, 'little', signed=True)
        elif isinstance(item, str):
            item = item.encode('utf-8')
        fp = self._fingerprint(item)
        if fp == 0:
            fp = 1
        i1 = z1hash64(item) % self._num_buckets
        i2 = (i1 ^ z1hash64(bytes([fp]))) % self._num_buckets
        for idx in (i1, i2):
            for s in range(self.BUCKET_SIZE):
                if self._table[idx][s] == fp:
                    self._table[idx][s] = 0
                    self._count -= 1
                    return True
        return False

    @property
    def count(self) -> int:
        return self._count

    @staticmethod
    def _fingerprint(item: bytes) -> int:
        """8 位指纹。"""
        return z1hash64(item, seed=0xDEADBEEF) & 0xFF
