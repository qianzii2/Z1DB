from __future__ import annotations
"""Ribbon Filter — 近最优空间概率过滤器。"""
from typing import List
from metal.hash import z1hash64


class RibbonFilter:
    __slots__ = ('_num_slots', '_width', '_data',
                 '_seeds', '_fp_bits')

    def __init__(self, keys: List[int],
                 fp_bits: int = 8,
                 width: int = 64) -> None:
        self._fp_bits = fp_bits; self._width = width
        n = len(keys)
        self._num_slots = max(int(n * 1.2) + 64, 128)
        self._data = [0] * self._num_slots
        self._seeds = [0xDEAD, 0xBEEF, 0xCAFE]
        self._build(keys)

    def contains(self, key: int) -> bool:
        start, coeff, fp = self._hash_key(key)
        result = 0
        for i in range(self._width):
            if coeff & (1 << i):
                idx = start + i
                if idx < self._num_slots:
                    result ^= self._data[idx]
        return (result & (
            (1 << self._fp_bits) - 1)) == fp

    def _build(self, keys):
        if not keys: return
        rows = []
        for key in keys:
            start, coeff, fp = self._hash_key(key)
            rows.append((start, coeff, fp))
        rows.sort(key=lambda r: r[0])
        pivot_row = {}
        eliminated = list(rows)
        for ri, (start, coeff, fp) in enumerate(eliminated):
            for col in range(self._width):
                actual_col = start + col
                if actual_col >= self._num_slots: break
                if not (coeff & (1 << col)): continue
                if actual_col in pivot_row:
                    ps, pc, pf = eliminated[
                        pivot_row[actual_col]]
                    shift = start - ps
                    aligned = (pc >> shift if shift >= 0
                               else pc << (-shift))
                    coeff ^= aligned; fp ^= pf
                    eliminated[ri] = (start, coeff, fp)
                else:
                    pivot_row[actual_col] = ri; break
        self._data = [0] * self._num_slots
        for ri in range(len(eliminated) - 1, -1, -1):
            start, coeff, fp = eliminated[ri]
            if coeff == 0: continue
            for col in range(self._width):
                if coeff & (1 << col):
                    actual_col = start + col
                    if actual_col >= self._num_slots: break
                    val = fp
                    for c2 in range(col + 1, self._width):
                        if coeff & (1 << c2):
                            idx = start + c2
                            if idx < self._num_slots:
                                val ^= self._data[idx]
                    self._data[actual_col] = val & (
                        (1 << self._fp_bits) - 1)
                    break

    def _hash_key(self, key):
        kb = key.to_bytes(8, 'little', signed=True)
        h1 = z1hash64(kb, seed=self._seeds[0])
        h2 = z1hash64(kb, seed=self._seeds[1])
        h3 = z1hash64(kb, seed=self._seeds[2])
        start = h1 % max(
            1, self._num_slots - self._width)
        coeff = h2 | 1
        coeff &= (1 << self._width) - 1
        fp = h3 & ((1 << self._fp_bits) - 1)
        if fp == 0: fp = 1
        return start, coeff, fp

    @property
    def size_bytes(self) -> int:
        return self._num_slots * 8
