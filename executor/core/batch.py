from __future__ import annotations
"""VectorBatch — a set of named DataVectors of equal length."""

from typing import Any, Dict, List, Optional

from executor.core.vector import DataVector
from metal.bitmap import Bitmap, BitmapLike
from metal.typed_vector import TypedVector
from storage.types import DTYPE_TO_ARRAY_CODE, DataType
from utils.errors import ColumnNotFoundError, ExecutionError


class VectorBatch:
    """A row-group of named, typed columns."""

    def __init__(self, columns: Dict[str, DataVector],
                 _row_count: int = 0,
                 _column_order: Optional[List[str]] = None) -> None:
        self.columns = columns
        self._row_count = _row_count
        self._column_order = _column_order or list(columns.keys())
        # Consistency check
        if columns:
            lengths = [len(v) for v in columns.values()]
            if lengths and any(l != lengths[0] for l in lengths):
                raise ExecutionError("VectorBatch column length mismatch")

    @property
    def row_count(self) -> int:
        if self.columns:
            return len(next(iter(self.columns.values())))
        return self._row_count

    @property
    def column_names(self) -> List[str]:
        return list(self._column_order)

    # ------------------------------------------------------------------
    def get_column(self, name: str) -> DataVector:
        if name not in self.columns:
            raise ColumnNotFoundError(name)
        return self.columns[name]

    def add_column(self, name: str, vec: DataVector) -> None:
        self.columns[name] = vec
        if name not in self._column_order:
            self._column_order.append(name)

    # ------------------------------------------------------------------
    def filter_by_bitmap(self, mask: BitmapLike) -> VectorBatch:
        indices = mask.to_indices()
        new_cols = {n: self.columns[n].filter_by_indices(indices) for n in self._column_order}
        return VectorBatch(new_cols, _row_count=len(indices), _column_order=list(self._column_order))

    def slice(self, start: int, end: int) -> VectorBatch:
        actual_end = min(end, self.row_count)
        indices = list(range(start, actual_end))
        new_cols = {n: self.columns[n].filter_by_indices(indices) for n in self._column_order}
        return VectorBatch(new_cols, _row_count=len(indices), _column_order=list(self._column_order))

    def reorder_by_indices(self, indices: List[int]) -> VectorBatch:
        new_cols = {n: self.columns[n].filter_by_indices(indices) for n in self._column_order}
        return VectorBatch(new_cols, _column_order=list(self._column_order))

    # ------------------------------------------------------------------
    def to_rows(self) -> List[list]:
        rc = self.row_count
        return [[self.columns[n].get(i) for n in self._column_order] for i in range(rc)]

    # ------------------------------------------------------------------
    @staticmethod
    def from_rows(rows: List[list], col_names: List[str],
                  col_types: List[DataType]) -> VectorBatch:
        if not rows:
            return VectorBatch.empty(col_names, col_types)
        cols: Dict[str, DataVector] = {}
        n = len(rows)
        for ci, (cname, ctype) in enumerate(zip(col_names, col_types)):
            nulls = Bitmap(n)
            code = DTYPE_TO_ARRAY_CODE.get(ctype)
            if code is not None:
                data: Any = TypedVector(code)
                for ri, row in enumerate(rows):
                    val = row[ci]
                    if val is None:
                        nulls.set_bit(ri)
                        data.append(0)
                    else:
                        data.append(val)
            elif ctype in (DataType.VARCHAR, DataType.TEXT):
                data = []
                for ri, row in enumerate(rows):
                    val = row[ci]
                    if val is None:
                        nulls.set_bit(ri)
                        data.append('')
                    else:
                        data.append(str(val))
            elif ctype == DataType.BOOLEAN:
                data = Bitmap(n)
                for ri, row in enumerate(rows):
                    val = row[ci]
                    if val is None:
                        nulls.set_bit(ri)
                    elif val:
                        data.set_bit(ri)
            else:
                data = TypedVector('q')
                for ri, row in enumerate(rows):
                    val = row[ci]
                    if val is None:
                        nulls.set_bit(ri)
                        data.append(0)
                    else:
                        data.append(val)
            cols[cname] = DataVector(dtype=ctype, data=data, nulls=nulls, _length=n)
        return VectorBatch(cols, _column_order=list(col_names))

    @staticmethod
    def empty(col_names: List[str], col_types: List[DataType]) -> VectorBatch:
        cols: Dict[str, DataVector] = {}
        for cname, ctype in zip(col_names, col_types):
            code = DTYPE_TO_ARRAY_CODE.get(ctype)
            if code is not None:
                data: Any = TypedVector(code)
            elif ctype in (DataType.VARCHAR, DataType.TEXT):
                data = []
            elif ctype == DataType.BOOLEAN:
                data = Bitmap(0)
            else:
                data = TypedVector('q')
            cols[cname] = DataVector(dtype=ctype, data=data, nulls=Bitmap(0), _length=0)
        return VectorBatch(cols, _row_count=0, _column_order=list(col_names))

    @staticmethod
    def single_row_no_columns() -> VectorBatch:
        return VectorBatch(columns={}, _row_count=1)

    @staticmethod
    def merge(batches: List[VectorBatch]) -> VectorBatch:
        if not batches:
            raise ExecutionError("cannot merge empty batch list")
        col_order = batches[0].column_names
        merged: Dict[str, DataVector] = {}
        for name in col_order:
            vecs = [b.columns[name] for b in batches]
            merged[name] = DataVector.concat(vecs)
        total = sum(b.row_count for b in batches)
        return VectorBatch(columns=merged, _row_count=total, _column_order=list(col_order))
