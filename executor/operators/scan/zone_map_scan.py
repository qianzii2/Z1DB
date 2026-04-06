from __future__ import annotations
"""ZoneMap 扫描 — 三级级联裁剪 + ColumnBloom 缓存。
[P18] ColumnBloomIndex 构建后缓存，跨 open() 调用复用。"""
from typing import Any, Dict, List, Optional, Tuple
from executor.core.batch import VectorBatch
from executor.core.operator import Operator
from executor.core.vector import DataVector
from parser.ast import BinaryExpr, ColumnRef, Literal
from storage.index.zone_map import ZoneMap, PruneResult
from storage.types import DataType

try:
    from storage.index.cascade_pruner import CascadePruner
    _HAS_CASCADE = True
except ImportError: _HAS_CASCADE = False
try:
    from structures.bloom_filter import BloomFilter
    _HAS_BLOOM = True
except ImportError: _HAS_BLOOM = False
try:
    from storage.index.column_bloom import ColumnBloomIndex
    _HAS_COL_BLOOM = True
except ImportError: _HAS_COL_BLOOM = False

# [P18] 模块级 ColumnBloom 缓存：(table_id, col_name, chunk_count) → ColumnBloomIndex
_COL_BLOOM_CACHE: Dict[tuple, 'ColumnBloomIndex'] = {}
_COL_BLOOM_CACHE_MAX = 64


class ZoneMapScanOperator(Operator):
    def __init__(self, table_name: str, store: Any,
                 columns: List[str], predicate: Any = None) -> None:
        super().__init__()
        self._table_name = table_name
        self._store = store
        self._columns = columns
        self._predicate = predicate
        self._chunk_indices: List[int] = []
        self._pos = 0

    def output_schema(self):
        col_map = {c.name: c.dtype for c in self._store.schema.columns}
        return [(n, col_map[n]) for n in self._columns]

    def open(self) -> None:
        total = self._store.get_chunk_count()
        if self._predicate:
            op, col, val = self._extract_simple_predicate(self._predicate)
            if op and col:
                self._chunk_indices = self._prune_chunks(total, col, op, val)
            else:
                self._chunk_indices = self._all_nonempty(total)
        else:
            self._chunk_indices = self._all_nonempty(total)
        self._pos = 0

    def _prune_chunks(self, total, col, op, val):
        chunks = self._store.get_column_chunks(col)
        nonempty = self._nonempty_indices(total)
        zone_maps = []
        bloom_filters: List[Optional[Any]] = []
        for ci in nonempty:
            if ci >= len(chunks): continue
            c = chunks[ci]
            zm = ZoneMap(min_val=c.zone_map.get('min'),
                         max_val=c.zone_map.get('max'),
                         null_count=c.zone_map.get('null_count', 0),
                         row_count=c.row_count)
            zone_maps.append(zm)
            bf = None
            if op == '=' and _HAS_BLOOM and c.dict_encoded is not None:
                bf = BloomFilter(max(c.dict_encoded.ndv, 1))
                for v in c.dict_encoded.dictionary:
                    bf.add(v)
            bloom_filters.append(bf)

        if _HAS_CASCADE and zone_maps:
            pruner = CascadePruner(zone_maps, bloom_filters)
            raw = pruner.prune(op, val)
            candidates = [nonempty[i] for i in raw if i < len(nonempty)]
        else:
            candidates = [nonempty[i] for i, zm in enumerate(zone_maps)
                          if zm.check(op, val) != PruneResult.ALWAYS_FALSE
                          and i < len(nonempty)]

        # [P18] 第三级 ColumnBloom 过滤（缓存版）
        if _HAS_COL_BLOOM and op == '=' and len(candidates) > 1:
            candidates = self._cached_column_bloom_filter(col, val, candidates)
        return candidates

    def _cached_column_bloom_filter(self, col, val, candidates):
        """[P18] 缓存 ColumnBloomIndex，避免每次 open 重建。"""
        global _COL_BLOOM_CACHE
        chunk_count = self._store.get_chunk_count()
        cache_key = (id(self._store), col, chunk_count)

        if cache_key not in _COL_BLOOM_CACHE:
            try:
                chunks = self._store.get_column_chunks(col)
                col_bloom = ColumnBloomIndex()
                all_nonempty = self._nonempty_indices(chunk_count)
                chunks_values = []
                for ci in all_nonempty:
                    if ci >= len(chunks):
                        chunks_values.append([])
                        continue
                    c = chunks[ci]
                    values = [c.get(i) for i in range(c.row_count)
                              if not c.null_bitmap.get_bit(i)]
                    chunks_values.append(values)
                col_bloom.build_for_column(col, chunks_values, fp_rate=0.01)
                # LRU 淘汰
                if len(_COL_BLOOM_CACHE) >= _COL_BLOOM_CACHE_MAX:
                    oldest = next(iter(_COL_BLOOM_CACHE))
                    del _COL_BLOOM_CACHE[oldest]
                _COL_BLOOM_CACHE[cache_key] = col_bloom
            except Exception:
                return candidates

        col_bloom = _COL_BLOOM_CACHE[cache_key]
        all_nonempty = self._nonempty_indices(chunk_count)
        # 候选 chunk 的索引需映射到 col_bloom 的内部索引
        nonempty_set = {ci: idx for idx, ci in enumerate(all_nonempty)}
        filtered = []
        for ci in candidates:
            bloom_idx = nonempty_set.get(ci)
            if bloom_idx is not None and col_bloom.check(col, bloom_idx, val):
                filtered.append(ci)
            elif bloom_idx is None:
                filtered.append(ci)  # 无映射则保留
        return filtered

    def _nonempty_indices(self, total):
        first_col = self._columns[0]
        chunks = self._store.get_column_chunks(first_col)
        return [ci for ci in range(min(total, len(chunks)))
                if chunks[ci].row_count > 0]

    def _all_nonempty(self, total):
        return self._nonempty_indices(total)

    def next_batch(self):
        if self._pos >= len(self._chunk_indices): return None
        ci = self._chunk_indices[self._pos]; self._pos += 1
        cols = {}
        for name in self._columns:
            cl = self._store.get_column_chunks(name)
            if ci < len(cl):
                cols[name] = DataVector.from_column_chunk(cl[ci])
        return VectorBatch(columns=cols, _column_order=list(self._columns)) if cols else None

    def close(self): pass

    @staticmethod
    def _extract_simple_predicate(pred):
        if isinstance(pred, BinaryExpr) and pred.op in ('>', '>=', '<', '<=', '=', '!='):
            if isinstance(pred.left, ColumnRef) and isinstance(pred.right, Literal):
                return pred.op, pred.left.column, pred.right.value
            if isinstance(pred.right, ColumnRef) and isinstance(pred.left, Literal):
                flip = {'>': '<', '<': '>', '>=': '<=', '<=': '>=', '=': '=', '!=': '!='}
                return flip.get(pred.op), pred.right.column, pred.left.value
        return None, None, None
