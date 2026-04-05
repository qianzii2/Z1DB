from __future__ import annotations
"""Per-column Bloom Filters for chunk-level filtering."""
from typing import Any, Dict, List
from structures.bloom_filter import BloomFilter


class ColumnBloomIndex:
    """Maintains Bloom Filters per column per chunk."""

    def __init__(self) -> None:
        self._filters: Dict[str, List[BloomFilter]] = {}

    def build_for_column(self, col_name: str, chunks_values: List[list],
                         fp_rate: float = 0.01) -> None:
        bfs = []
        for values in chunks_values:
            non_null = [v for v in values if v is not None]
            bf = BloomFilter(max(len(non_null), 1), fp_rate)
            for v in non_null:
                bf.add(v)
            bfs.append(bf)
        self._filters[col_name] = bfs

    def check(self, col_name: str, chunk_idx: int, value: Any) -> bool:
        if col_name not in self._filters:
            return True
        bfs = self._filters[col_name]
        if chunk_idx >= len(bfs):
            return True
        return bfs[chunk_idx].contains(value)

    def prune_chunks(self, col_name: str, value: Any) -> List[int]:
        if col_name not in self._filters:
            return list(range(len(self._filters.get(col_name, []))))
        return [i for i, bf in enumerate(self._filters[col_name])
                if bf.contains(value)]
