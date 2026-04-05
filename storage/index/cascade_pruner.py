from __future__ import annotations
"""Cascade Pruner — ZoneMap → BloomFilter → precise scan. 3-stage filter."""
from typing import Any, List, Optional
from storage.index.zone_map import ZoneMap, PruneResult

try:
    from structures.bloom_filter import BloomFilter
    _HAS_BLOOM = True
except ImportError:
    _HAS_BLOOM = False


class CascadePruner:
    """Three-stage cascade filter for chunk pruning.
    Stage 1: Zone Map (O(1) per chunk) → exclude by value range
    Stage 2: Bloom Filter (O(k) per chunk) → exclude by set membership
    Stage 3: Precise scan (only surviving chunks)
    Typical: 1000 chunks → ZM:100 → BF:10 → scan 10."""

    def __init__(self, zone_maps: List[ZoneMap],
                 bloom_filters: Optional[List[Optional[BloomFilter]]] = None) -> None:
        self._zone_maps = zone_maps
        self._blooms = bloom_filters

    def prune(self, op: str, value: Any) -> List[int]:
        """Return chunk indices that might contain matching rows."""
        # Stage 1: Zone Map
        candidates = []
        for i, zm in enumerate(self._zone_maps):
            result = zm.check(op, value)
            if result != PruneResult.ALWAYS_FALSE:
                candidates.append(i)

        # Stage 2: Bloom Filter (only for equality)
        if op == '=' and self._blooms and _HAS_BLOOM:
            bf_candidates = []
            for i in candidates:
                if i < len(self._blooms) and self._blooms[i] is not None:
                    if self._blooms[i].contains(value):
                        bf_candidates.append(i)
                else:
                    bf_candidates.append(i)  # No BF → keep
            candidates = bf_candidates

        return candidates

    def prune_range(self, lo: Any, hi: Any) -> List[int]:
        """Prune for BETWEEN lo AND hi."""
        candidates = []
        for i, zm in enumerate(self._zone_maps):
            if zm.min_val is None:
                candidates.append(i); continue
            try:
                if zm.max_val < lo or zm.min_val > hi:
                    continue
            except TypeError:
                pass
            candidates.append(i)
        return candidates

    @property
    def total_chunks(self) -> int:
        return len(self._zone_maps)

    def pruned_ratio(self, surviving: List[int]) -> float:
        total = self.total_chunks
        if total == 0: return 0.0
        return 1.0 - len(surviving) / total
