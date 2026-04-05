from __future__ import annotations
"""ZoneMap scan — skips chunks that cannot contain matching rows."""
from typing import Any, List, Optional, Tuple
from executor.core.batch import VectorBatch
from executor.core.operator import Operator
from executor.core.vector import DataVector
from executor.expression.evaluator import ExpressionEvaluator
from parser.ast import BinaryExpr, ColumnRef, Literal
from storage.index.zone_map import ZoneMap, PruneResult
from storage.table_store import TableStore
from storage.types import DataType


class ZoneMapScanOperator(Operator):
    """SeqScan with ZoneMap chunk pruning."""

    def __init__(self, table_name: str, store: TableStore,
                 columns: List[str], predicate: Any = None) -> None:
        super().__init__()
        self._store = store; self._columns = columns
        self._predicate = predicate
        self._chunk_indices: List[int] = []
        self._pos = 0

    def output_schema(self) -> List[Tuple[str, DataType]]:
        col_map = {c.name: c.dtype for c in self._store.schema.columns}
        return [(n, col_map[n]) for n in self._columns]

    def open(self) -> None:
        total = self._store.get_chunk_count()
        # Determine which chunks to scan
        if self._predicate:
            op, col, val = self._extract_simple_predicate(self._predicate)
            if op and col:
                self._chunk_indices = []
                for ci in range(total):
                    chunks = self._store.get_column_chunks(col)
                    if ci >= len(chunks):
                        continue
                    chunk = chunks[ci]
                    if chunk.row_count == 0:
                        continue
                    zm = ZoneMap(min_val=chunk.zone_map.get('min'),
                                max_val=chunk.zone_map.get('max'),
                                null_count=chunk.zone_map.get('null_count', 0),
                                row_count=chunk.row_count)
                    result = zm.check(op, val)
                    if result != PruneResult.ALWAYS_FALSE:
                        self._chunk_indices.append(ci)
            else:
                self._chunk_indices = [i for i in range(total)]
        else:
            self._chunk_indices = [i for i in range(total)]
        # Filter out empty chunks
        self._chunk_indices = [
            ci for ci in self._chunk_indices
            if ci < total and self._store.get_column_chunks(
                self._columns[0])[ci].row_count > 0]
        self._pos = 0

    def next_batch(self) -> Optional[VectorBatch]:
        if self._pos >= len(self._chunk_indices):
            return None
        ci = self._chunk_indices[self._pos]
        self._pos += 1
        cols = {}
        for name in self._columns:
            chunk_list = self._store.get_column_chunks(name)
            cols[name] = DataVector.from_column_chunk(chunk_list[ci])
        return VectorBatch(columns=cols, _column_order=list(self._columns))

    def close(self) -> None: pass

    @staticmethod
    def _extract_simple_predicate(pred: Any) -> Tuple[Optional[str], Optional[str], Any]:
        """Extract (op, column_name, value) from simple predicates."""
        if isinstance(pred, BinaryExpr) and pred.op in ('>', '>=', '<', '<=', '=', '!='):
            col = val = None
            if isinstance(pred.left, ColumnRef) and isinstance(pred.right, Literal):
                col = pred.left.column; val = pred.right.value
            elif isinstance(pred.right, ColumnRef) and isinstance(pred.left, Literal):
                col = pred.right.column; val = pred.left.value
                # Flip operator
                flip = {'>':'<','<':'>','>=':'<=','<=':'>=','=':'=','!=':'!='}
                return flip.get(pred.op), col, val
            return pred.op, col, val
        return None, None, None
