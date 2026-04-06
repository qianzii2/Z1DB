from __future__ import annotations
"""MinHash + LSH — Jaccard 相似度近似。
jaccard_exact / cosine_similarity 是全项目唯一的精确实现。
evaluator 中的 _eval_jaccard/_eval_cosine 应委托到此处。"""
import random
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple
from metal.hash import z1hash64


class MinHash:
    """MinHash 签名生成器。"""
    __slots__ = ('_num_hashes', '_seeds')

    def __init__(self, num_hashes: int = 128,
                 seed: int = 42) -> None:
        self._num_hashes = num_hashes
        rng = random.Random(seed)
        self._seeds = [rng.getrandbits(64)
                       for _ in range(num_hashes)]

    def signature(self, elements: set) -> List[int]:
        sig = [0x7FFFFFFFFFFFFFFF] * self._num_hashes
        for elem in elements:
            if isinstance(elem, str):
                elem_bytes = elem.encode('utf-8')
            elif isinstance(elem, int):
                elem_bytes = elem.to_bytes(8, 'little', signed=True)
            else:
                elem_bytes = str(elem).encode('utf-8')
            for i in range(self._num_hashes):
                h = z1hash64(elem_bytes, seed=self._seeds[i])
                if h < sig[i]:
                    sig[i] = h
        return sig

    def jaccard_estimate(self, sig_a: List[int],
                         sig_b: List[int]) -> float:
        if len(sig_a) != len(sig_b):
            raise ValueError("签名长度必须相同")
        matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
        return matches / len(sig_a)


class LSH:
    """Locality-Sensitive Hashing。"""
    __slots__ = ('_bands', '_rows', '_buckets', '_minhash')

    def __init__(self, num_hashes: int = 128,
                 bands: int = 16, seed: int = 42) -> None:
        self._rows = num_hashes // bands
        self._bands = bands
        self._minhash = MinHash(
            num_hashes=bands * self._rows, seed=seed)
        self._buckets: List[Dict[tuple, List[Any]]] = [
            {} for _ in range(bands)]

    @property
    def threshold(self) -> float:
        return (1.0 / self._bands) ** (1.0 / self._rows)

    def index(self, item_id: Any, elements: set) -> None:
        sig = self._minhash.signature(elements)
        for band in range(self._bands):
            start = band * self._rows
            band_sig = tuple(sig[start:start + self._rows])
            self._buckets[band].setdefault(band_sig, []).append(item_id)

    def query(self, elements: set) -> Set[Any]:
        sig = self._minhash.signature(elements)
        candidates: Set[Any] = set()
        for band in range(self._bands):
            start = band * self._rows
            band_sig = tuple(sig[start:start + self._rows])
            if band_sig in self._buckets[band]:
                candidates.update(self._buckets[band][band_sig])
        return candidates

    def find_similar_pairs(
            self, items: Dict[Any, set],
            min_jaccard: float = 0.5
    ) -> List[Tuple[Any, Any, float]]:
        sigs: Dict[Any, List[int]] = {}
        for item_id, elements in items.items():
            sig = self._minhash.signature(elements)
            sigs[item_id] = sig
            self.index(item_id, elements)
        seen: set = set()
        results: List[Tuple[Any, Any, float]] = []
        for item_id, elements in items.items():
            candidates = self.query(elements)
            for cand_id in candidates:
                if cand_id == item_id:
                    continue
                pair = (min(item_id, cand_id), max(item_id, cand_id))
                if pair in seen:
                    continue
                seen.add(pair)
                j = self._minhash.jaccard_estimate(
                    sigs[item_id], sigs[cand_id])
                if j >= min_jaccard:
                    results.append((pair[0], pair[1], j))
        return results


# ═══ 精确相似度函数（全项目唯一实现）═══

def jaccard_exact(set_a: set, set_b: set) -> float:
    """精确 Jaccard 相似度。evaluator._eval_jaccard 委托到此。"""
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def cosine_similarity(vec_a: List[float],
                      vec_b: List[float]) -> float:
    """精确余弦相似度。evaluator._eval_cosine 委托到此。"""
    min_len = min(len(vec_a), len(vec_b))
    if min_len == 0:
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a[:min_len], vec_b[:min_len]))
    mag_a = sum(a * a for a in vec_a[:min_len]) ** 0.5
    mag_b = sum(b * b for b in vec_b[:min_len]) ** 0.5
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)
