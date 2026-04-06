from __future__ import annotations
"""列级 Bloom Filter — 每列每 chunk 一个 Bloom Filter。
用于 ZoneMapScan 的第三级裁剪：ZoneMap → CascadePruner → ColumnBloom。"""
from typing import Any, Dict, List
from structures.bloom_filter import BloomFilter


class ColumnBloomIndex:
    """维护每列每 chunk 的 Bloom Filter。"""

    def __init__(self) -> None:
        self._filters: Dict[str, List[BloomFilter]] = {}

    def build_for_column(self, col_name: str,
                         chunks_values: List[list],
                         fp_rate: float = 0.01) -> None:
        """为指定列的所有 chunk 构建 Bloom Filter。"""
        bfs = []
        for values in chunks_values:
            non_null = [v for v in values if v is not None]
            bf = BloomFilter(max(len(non_null), 1), fp_rate)
            for v in non_null:
                bf.add(v)
            bfs.append(bf)
        self._filters[col_name] = bfs

    def check(self, col_name: str, chunk_idx: int,
              value: Any) -> bool:
        """检查值是否可能在指定列的指定 chunk 中。
        True = 可能在，False = 一定不在。"""
        if col_name not in self._filters:
            return True  # 无 filter → 保守返回 True
        bfs = self._filters[col_name]
        if chunk_idx >= len(bfs):
            return True
        return bfs[chunk_idx].contains(value)

    def prune_chunks(self, col_name: str,
                     value: Any) -> List[int]:
        """返回可能包含 value 的 chunk 索引列表。"""
        if col_name not in self._filters:
            return list(range(
                len(self._filters.get(col_name, []))))
        return [i for i, bf in enumerate(
            self._filters[col_name])
            if bf.contains(value)]
