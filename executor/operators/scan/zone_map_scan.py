from __future__ import annotations
"""ZoneMap扫描 — 跳过不含匹配行的chunk。集成CascadePruner三级裁剪。"""
from typing import Any, List, Optional, Tuple
from executor.core.batch import VectorBatch
from executor.core.operator import Operator
from executor.core.vector import DataVector
from executor.expression.evaluator import ExpressionEvaluator
from parser.ast import BinaryExpr, ColumnRef, Literal
from storage.index.zone_map import ZoneMap, PruneResult
from storage.table_store import TableStore
from storage.types import DataType

try:
    from storage.index.cascade_pruner import CascadePruner
    _HAS_CASCADE = True
except ImportError:
    _HAS_CASCADE = False

try:
    from structures.bloom_filter import BloomFilter
    _HAS_BLOOM = True
except ImportError:
    _HAS_BLOOM = False


class ZoneMapScanOperator(Operator):
    """顺序扫描 + ZoneMap chunk裁剪。可选Bloom Filter二级过滤。"""

    def __init__(self, table_name: str, store: TableStore,
                 columns: List[str], predicate: Any = None) -> None:
        super().__init__()
        self._store = store
        self._columns = columns
        self._predicate = predicate
        self._chunk_indices: List[int] = []
        self._pos = 0

    def output_schema(self) -> List[Tuple[str, DataType]]:
        col_map = {c.name: c.dtype for c in self._store.schema.columns}
        return [(n, col_map[n]) for n in self._columns]

    def open(self) -> None:
        total = self._store.get_chunk_count()
        if self._predicate:
            op, col, val = self._extract_simple_predicate(self._predicate)
            if op and col:
                self._chunk_indices = self._prune_chunks(
                    total, col, op, val)
            else:
                self._chunk_indices = self._all_nonempty(total)
        else:
            self._chunk_indices = self._all_nonempty(total)
        self._pos = 0

    def _prune_chunks(self, total: int, col: str, op: str,
                      val: Any) -> List[int]:
        """用CascadePruner或简单ZoneMap裁剪。"""
        chunks = self._store.get_column_chunks(col)

        # 构建ZoneMap列表
        zone_maps = []
        bloom_filters: List[Optional[Any]] = []
        for ci in range(total):
            if ci >= len(chunks):
                continue
            c = chunks[ci]
            if c.row_count == 0:
                continue
            zm = ZoneMap(
                min_val=c.zone_map.get('min'),
                max_val=c.zone_map.get('max'),
                null_count=c.zone_map.get('null_count', 0),
                row_count=c.row_count)
            zone_maps.append(zm)

            # 等值查询时，如果有字典编码可构建Bloom
            bf = None
            if op == '=' and _HAS_BLOOM and c.dict_encoded is not None:
                bf = BloomFilter(max(c.dict_encoded.ndv, 1))
                for v in c.dict_encoded.dictionary:
                    bf.add(v)
            bloom_filters.append(bf)

        # 用CascadePruner三级裁剪
        if _HAS_CASCADE and zone_maps:
            pruner = CascadePruner(zone_maps, bloom_filters)
            raw_indices = pruner.prune(op, val)
            # raw_indices是zone_maps列表中的索引，需映射回实际chunk索引
            nonempty = self._nonempty_indices(total)
            return [nonempty[i] for i in raw_indices
                    if i < len(nonempty)]

        # 回退简单ZoneMap裁剪
        result = []
        nonempty = self._nonempty_indices(total)
        for idx, zm in enumerate(zone_maps):
            if zm.check(op, val) != PruneResult.ALWAYS_FALSE:
                if idx < len(nonempty):
                    result.append(nonempty[idx])
        return result

    def _nonempty_indices(self, total: int) -> List[int]:
        """返回所有非空chunk的索引。"""
        result = []
        first_col = self._columns[0]
        chunks = self._store.get_column_chunks(first_col)
        for ci in range(min(total, len(chunks))):
            if chunks[ci].row_count > 0:
                result.append(ci)
        return result

    def _all_nonempty(self, total: int) -> List[int]:
        return self._nonempty_indices(total)

    def next_batch(self) -> Optional[VectorBatch]:
        if self._pos >= len(self._chunk_indices):
            return None
        ci = self._chunk_indices[self._pos]
        self._pos += 1
        cols = {}
        for name in self._columns:
            chunk_list = self._store.get_column_chunks(name)
            if ci < len(chunk_list):
                cols[name] = DataVector.from_column_chunk(chunk_list[ci])
        if not cols:
            return None
        return VectorBatch(columns=cols, _column_order=list(self._columns))

    def close(self) -> None:
        pass

    @staticmethod
    def _extract_simple_predicate(
            pred: Any) -> Tuple[Optional[str], Optional[str], Any]:
        """从简单谓词中提取 (op, column_name, value)。"""
        if (isinstance(pred, BinaryExpr)
                and pred.op in ('>', '>=', '<', '<=', '=', '!=')):
            col = val = None
            if (isinstance(pred.left, ColumnRef)
                    and isinstance(pred.right, Literal)):
                col = pred.left.column
                val = pred.right.value
                return pred.op, col, val
            elif (isinstance(pred.right, ColumnRef)
                  and isinstance(pred.left, Literal)):
                col = pred.right.column
                val = pred.left.value
                flip = {'>': '<', '<': '>', '>=': '<=', '<=': '>=',
                        '=': '=', '!=': '!='}
                return flip.get(pred.op), col, val
            return pred.op, col, val
        return None, None, None
