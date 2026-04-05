from __future__ import annotations

"""MinHash + LSH — Jaccard similarity estimation + approximate nearest neighbor.
Paper: Broder, 1997 (MinHash) + Indyk & Motwani, 1998 (LSH).
O(1) per-pair Jaccard estimate after O(n) preprocessing."""
import random
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple
from metal.hash import murmur3_64


class MinHash:
    """MinHash signature for Jaccard similarity estimation."""

    __slots__ = ('_num_hashes', '_seeds')

    def __init__(self, num_hashes: int = 128, seed: int = 42) -> None:
        self._num_hashes = num_hashes
        rng = random.Random(seed)
        self._seeds = [rng.getrandbits(64) for _ in range(num_hashes)]

    def signature(self, elements: set) -> List[int]:
        """Compute MinHash signature for a set. O(|set| × num_hashes)."""
        sig = [0x7FFFFFFFFFFFFFFF] * self._num_hashes
        for elem in elements:
            if isinstance(elem, str):
                elem_bytes = elem.encode('utf-8')
            elif isinstance(elem, int):
                elem_bytes = elem.to_bytes(8, 'little', signed=True)
            else:
                elem_bytes = str(elem).encode('utf-8')
            for i in range(self._num_hashes):
                h = murmur3_64(elem_bytes, seed=self._seeds[i])
                if h < sig[i]:
                    sig[i] = h
        return sig

    def jaccard_estimate(self, sig_a: List[int], sig_b: List[int]) -> float:
        """Estimate Jaccard similarity from two signatures. O(num_hashes)."""
        if len(sig_a) != len(sig_b):
            raise ValueError("signature lengths must match")
        matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
        return matches / len(sig_a)


class LSH:
    """Locality-Sensitive Hashing for approximate nearest neighbor search.

    Splits signature into `bands` bands of `rows` rows each.
    Threshold ≈ (1/bands)^(1/rows).
    """

    __slots__ = ('_bands', '_rows', '_buckets', '_minhash')

    def __init__(self, num_hashes: int = 128, bands: int = 16,
                 seed: int = 42) -> None:
        self._rows = num_hashes // bands
        self._bands = bands
        self._minhash = MinHash(num_hashes=bands * self._rows, seed=seed)
        self._buckets: List[Dict[tuple, List[Any]]] = [
            {} for _ in range(bands)]

    @property
    def threshold(self) -> float:
        """Approximate Jaccard threshold for candidate pairs."""
        return (1.0 / self._bands) ** (1.0 / self._rows)

    def index(self, item_id: Any, elements: set) -> None:
        """Add an item to the index."""
        sig = self._minhash.signature(elements)
        for band in range(self._bands):
            start = band * self._rows
            band_sig = tuple(sig[start:start + self._rows])
            if band_sig not in self._buckets[band]:
                self._buckets[band][band_sig] = []
            self._buckets[band][band_sig].append(item_id)

    def query(self, elements: set) -> Set[Any]:
        """Find candidate similar items. O(bands × rows)."""
        sig = self._minhash.signature(elements)
        candidates: Set[Any] = set()
        for band in range(self._bands):
            start = band * self._rows
            band_sig = tuple(sig[start:start + self._rows])
            if band_sig in self._buckets[band]:
                candidates.update(self._buckets[band][band_sig])
        return candidates

    def find_similar_pairs(self, items: Dict[Any, set],
                           min_jaccard: float = 0.5) -> List[Tuple[Any, Any, float]]:
        """Find all pairs with Jaccard ≥ min_jaccard. Approximate."""
        # Index all items
        sigs: Dict[Any, List[int]] = {}
        for item_id, elements in items.items():
            sig = self._minhash.signature(elements)
            sigs[item_id] = sig
            self.index(item_id, elements)

        # Find candidate pairs
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
                # Exact Jaccard from signatures
                j = self._minhash.jaccard_estimate(sigs[item_id], sigs[cand_id])
                if j >= min_jaccard:
                    results.append((pair[0], pair[1], j))

        return results


def jaccard_exact(set_a: set, set_b: set) -> float:
    """Exact Jaccard similarity. O(|A| + |B|)."""
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Exact cosine similarity. O(n)."""
    if len(vec_a) != len(vec_b):
        raise ValueError("vectors must have same length")
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = sum(a * a for a in vec_a) ** 0.5
    mag_b = sum(b * b for b in vec_b) ** 0.5
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)
