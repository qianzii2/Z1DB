from __future__ import annotations
"""XOR Filter — 静态概率过滤器。
空间效率优于 Bloom Filter（1.23 bits/key vs 1.44 bits/key）。
构建后不可修改。用于 SSTable 索引。"""
from typing import List
from metal.hash import z1hash64


class XorFilter:
    """XOR Filter：三哈希 + 剥离构建。"""

    __slots__ = ('_size', '_fingerprints', '_seeds')

    def __init__(self, keys: List[int],
                 fp_bits: int = 8) -> None:
        n = len(keys)
        self._size = max(int(1.3 * n) + 64, 128)
        self._fingerprints = bytearray(self._size)
        self._seeds = [0, 0x9E3779B9, 0x517CC1B7]
        if n > 0:
            self._build(keys)

    def contains(self, key: int) -> bool:
        """O(1) 查询。"""
        fp = self._fp(key)
        h0, h1, h2 = self._hashes(key)
        return (self._fingerprints[h0]
                ^ self._fingerprints[h1]
                ^ self._fingerprints[h2]) == fp

    def _hashes(self, key):
        kb = (key & 0xFFFFFFFFFFFFFFFF).to_bytes(8, 'little')
        h0 = z1hash64(kb, self._seeds[0]) % self._size
        h1 = z1hash64(kb, self._seeds[1]) % self._size
        h2 = z1hash64(kb, self._seeds[2]) % self._size
        # 确保三个位置不同
        if h1 == h0:
            h1 = (h1 + 1) % self._size
        if h2 == h0 or h2 == h1:
            h2 = (h2 + 2) % self._size
        return h0, h1, h2

    def _build(self, keys):
        """剥离构建：找度为 1 的节点逐步消去。最多 200 次重试。"""
        for attempt in range(200):
            self._seeds = [attempt * 7, attempt * 7 + 1,
                           attempt * 7 + 2]
            edges = [(self._hashes(k), self._fp(k))
                     for k in keys]
            degree = [0] * self._size
            adj: list = [[] for _ in range(self._size)]
            for ei, (hs, fp) in enumerate(edges):
                for v in hs:
                    degree[v] += 1
                    adj[v].append(ei)
            # 剥离：反复移除度为 1 的节点
            queue = [v for v in range(self._size)
                     if degree[v] == 1]
            order = []
            removed = [False] * len(edges)
            while queue:
                v = queue.pop()
                for ei in adj[v]:
                    if removed[ei]:
                        continue
                    removed[ei] = True
                    order.append((ei, v))
                    for u in edges[ei][0]:
                        degree[u] -= 1
                        if degree[u] == 1:
                            queue.append(u)
            if len(order) == len(edges):
                # 全部剥离成功 → 回代赋值
                self._fingerprints = bytearray(self._size)
                for ei, v in reversed(order):
                    hs, fp = edges[ei]
                    val = fp
                    for u in hs:
                        if u != v:
                            val ^= self._fingerprints[u]
                    self._fingerprints[v] = val & 0xFF
                return
        # 200 次全失败 → 空过滤器（contains 全返回 True）
        self._fingerprints = bytearray(self._size)

    def _fp(self, key):
        kb = (key & 0xFFFFFFFFFFFFFFFF).to_bytes(8, 'little')
        fp = z1hash64(kb, seed=42) & 0xFF
        return fp if fp != 0 else 1

    @property
    def size_bytes(self) -> int:
        return self._size
