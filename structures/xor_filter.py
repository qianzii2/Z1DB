from __future__ import annotations
"""XOR Filter — static probabilistic filter."""
from typing import List
from metal.hash import murmur3_64


class XorFilter:
    __slots__ = ('_size', '_fingerprints', '_seeds')

    def __init__(self, keys: List[int], fp_bits: int = 8) -> None:
        n = len(keys)
        self._size = max(int(1.3 * n) + 64, 128)
        self._fingerprints = bytearray(self._size)
        self._seeds = [0, 0x9E3779B9, 0x517CC1B7]
        if n > 0:
            self._build(keys)

    def contains(self, key: int) -> bool:
        fp = self._fp(key)
        h0, h1, h2 = self._hashes(key)
        return (self._fingerprints[h0] ^ self._fingerprints[h1] ^ self._fingerprints[h2]) == fp

    def _hashes(self, key: int) -> tuple:
        kb = (key & 0xFFFFFFFFFFFFFFFF).to_bytes(8, 'little')
        h0 = murmur3_64(kb, self._seeds[0]) % self._size
        h1 = murmur3_64(kb, self._seeds[1]) % self._size
        h2 = murmur3_64(kb, self._seeds[2]) % self._size
        # Ensure distinct positions
        if h1 == h0: h1 = (h1 + 1) % self._size
        if h2 == h0 or h2 == h1: h2 = (h2 + 2) % self._size
        return h0, h1, h2

    def _build(self, keys: List[int]) -> None:
        for attempt in range(200):
            self._seeds = [attempt * 7, attempt * 7 + 1, attempt * 7 + 2]
            edges = [(self._hashes(k), self._fp(k)) for k in keys]
            degree = [0] * self._size
            adj: list = [[] for _ in range(self._size)]
            for ei, (hs, fp) in enumerate(edges):
                for v in hs:
                    degree[v] += 1
                    adj[v].append(ei)
            queue = [v for v in range(self._size) if degree[v] == 1]
            order = []
            removed = [False] * len(edges)
            while queue:
                v = queue.pop()
                for ei in adj[v]:
                    if removed[ei]: continue
                    removed[ei] = True
                    order.append((ei, v))
                    for u in edges[ei][0]:
                        degree[u] -= 1
                        if degree[u] == 1: queue.append(u)
            if len(order) == len(edges):
                self._fingerprints = bytearray(self._size)
                for ei, v in reversed(order):
                    hs, fp = edges[ei]
                    val = fp
                    for u in hs:
                        if u != v: val ^= self._fingerprints[u]
                    self._fingerprints[v] = val & 0xFF
                return
        # Fallback: brute force store
        self._fingerprints = bytearray(self._size)

    def _fp(self, key: int) -> int:
        kb = (key & 0xFFFFFFFFFFFFFFFF).to_bytes(8, 'little')
        fp = murmur3_64(kb, seed=42) & 0xFF
        return fp if fp != 0 else 1

    @property
    def size_bytes(self) -> int:
        return self._size
