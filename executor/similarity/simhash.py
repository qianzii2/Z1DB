from __future__ import annotations
"""SimHash — 余弦相似度近似。"""
from typing import List
from metal.hash import z1hash64
from metal.bitwise import popcount64


def simhash(tokens: List[str], dim: int = 64) -> int:
    v = [0] * dim
    for token in tokens:
        h = z1hash64(token.encode('utf-8'))
        for i in range(dim):
            if h & (1 << i): v[i] += 1
            else: v[i] -= 1
    result = 0
    for i in range(dim):
        if v[i] > 0: result |= (1 << i)
    return result


def hamming_distance(a: int, b: int) -> int:
    return popcount64(a ^ b)


def simhash_similarity(a: int, b: int,
                       dim: int = 64) -> float:
    return 1.0 - hamming_distance(a, b) / dim


def text_similarity(text_a: str,
                    text_b: str) -> float:
    tokens_a = text_a.lower().split()
    tokens_b = text_b.lower().split()
    sig_a = simhash(tokens_a)
    sig_b = simhash(tokens_b)
    return simhash_similarity(sig_a, sig_b)
