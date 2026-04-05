from __future__ import annotations
"""索引扫描 — 用ART/SkipList做点查/范围查。
通过read_rows_by_indices避免全表读取。"""
from typing import Any, Dict, List, Optional, Tuple
from executor.core.batch import VectorBatch
from executor.core.operator import Operator
from executor.core.vector import DataVector
from storage.table_store import TableStore
from storage.types import DataType

try:
    from structures.art import AdaptiveRadixTree
    _HAS_ART = True
except ImportError:
    _HAS_ART = False

try:
    from structures.skip_list import SkipList
    _HAS_SKIP = True
except ImportError:
    _HAS_SKIP = False


class IndexScanOperator(Operator):
    """用索引做点查或范围查。"""

    def __init__(self, table_name: str, store: TableStore,
                 columns: List[str], index: Any,
                 scan_type: str = 'EQ',
                 key_value: Any = None,
                 range_lo: Any = None,
                 range_hi: Any = None) -> None:
        super().__init__()
        self._store = store
        self._columns = columns
        self._index = index
        self._scan_type = scan_type
        self._key_value = key_value
        self._range_lo = range_lo
        self._range_hi = range_hi
        self._result_rows: list = []
        self._emitted = False

    def output_schema(self) -> List[Tuple[str, DataType]]:
        col_map = {c.name: c.dtype for c in self._store.schema.columns}
        return [(n, col_map[n]) for n in self._columns]

    def open(self) -> None:
        self._result_rows = []
        row_ids: List[int] = []

        if self._scan_type == 'EQ' and self._key_value is not None:
            if _HAS_ART and isinstance(self._index, AdaptiveRadixTree):
                key = str(self._key_value).encode('utf-8')
                result = self._index.search(key)
                if result is not None:
                    if isinstance(result, list):
                        row_ids.extend(result)
                    else:
                        row_ids.append(result)
            elif _HAS_SKIP and isinstance(self._index, SkipList):
                result = self._index.search(self._key_value)
                if result is not None:
                    if isinstance(result, list):
                        row_ids.extend(result)
                    else:
                        row_ids.append(result)

        elif self._scan_type == 'RANGE':
            if _HAS_SKIP and isinstance(self._index, SkipList):
                results = self._index.range_query(
                    self._range_lo, self._range_hi)
                for _, rid in results:
                    if isinstance(rid, list):
                        row_ids.extend(rid)
                    else:
                        row_ids.append(rid)

        elif self._scan_type == 'PREFIX' and _HAS_ART:
            if isinstance(self._index, AdaptiveRadixTree):
                prefix = str(self._key_value).encode('utf-8')
                results = self._index.prefix_scan(prefix)
                for _, rid in results:
                    if isinstance(rid, list):
                        row_ids.extend(rid)
                    else:
                        row_ids.append(rid)

        # 用read_rows_by_indices避免全表读取
        if row_ids:
            valid_ids = [rid for rid in row_ids
                         if 0 <= rid < self._store.row_count]
            fetched = self._store.read_rows_by_indices(valid_ids)
            col_names = self._store.schema.column_names
            col_indices = [col_names.index(c) for c in self._columns
                           if c in col_names]
            for full_row in fetched:
                row = [full_row[ci] for ci in col_indices]
                self._result_rows.append(row)

        self._emitted = False

    def next_batch(self) -> Optional[VectorBatch]:
        if self._emitted:
            return None
        self._emitted = True
        schema = self.output_schema()
        names = [n for n, _ in schema]
        types = [t for _, t in schema]
        if not self._result_rows:
            return VectorBatch.empty(names, types)
        return VectorBatch.from_rows(self._result_rows, names, types)

    def close(self) -> None:
        pass
