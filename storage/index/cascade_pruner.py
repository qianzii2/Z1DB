from __future__ import annotations
"""级联裁剪器 — ZoneMap → BloomFilter → 精确扫描。三级过滤。
典型效果：1000 chunks → ZoneMap 保留 100 → Bloom 保留 10 → 扫描 10。"""
from typing import Any, List, Optional
from storage.index.zone_map import ZoneMap, PruneResult

try:
    from structures.bloom_filter import BloomFilter
    _HAS_BLOOM = True
except ImportError:
    _HAS_BLOOM = False


class CascadePruner:
    """三级级联裁剪。每级淘汰不可能命中的 chunk。"""

    def __init__(self, zone_maps: List[ZoneMap],
                 bloom_filters: Optional[List[Optional[BloomFilter]]] = None
                 ) -> None:
        self._zone_maps = zone_maps
        self._blooms = bloom_filters

    def prune(self, op: str, value: Any) -> List[int]:
        """返回可能包含匹配行的 chunk 索引。"""
        # 第一级：ZoneMap（O(1) 每 chunk）
        candidates = []
        for i, zm in enumerate(self._zone_maps):
            if zm.check(op, value) != PruneResult.ALWAYS_FALSE:
                candidates.append(i)

        # 第二级：Bloom Filter（仅等值查询）
        if op == '=' and self._blooms and _HAS_BLOOM:
            bf_candidates = []
            for i in candidates:
                if (i < len(self._blooms)
                        and self._blooms[i] is not None):
                    if self._blooms[i].contains(value):
                        bf_candidates.append(i)
                else:
                    bf_candidates.append(i)  # 无 BF → 保留
            candidates = bf_candidates

        return candidates

    def prune_range(self, lo: Any, hi: Any) -> List[int]:
        """范围裁剪：BETWEEN lo AND hi。"""
        candidates = []
        for i, zm in enumerate(self._zone_maps):
            if zm.min_val is None:
                candidates.append(i)
                continue
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
        """计算裁剪率。"""
        total = self.total_chunks
        if total == 0:
            return 0.0
        return 1.0 - len(surviving) / total
