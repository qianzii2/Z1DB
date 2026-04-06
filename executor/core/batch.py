from __future__ import annotations
"""VectorBatch — 向量化批处理容器。
[P01] from_rows 列优先批量构建，TypedVector 使用初始化列表一次性构建。
[D03] 移除 from_rows 中的 Arena（Arena 适合算子级而非临时操作）。"""
from typing import Any, Dict, List, Optional

from executor.core.vector import (
    DataVector, _pack_values, _unpack_to_typed, _NANBOX_TYPES)
from metal.bitmap import Bitmap, BitmapLike
from metal.typed_vector import TypedVector
from storage.types import DTYPE_TO_ARRAY_CODE, DataType
from utils.errors import ColumnNotFoundError, ExecutionError

try:
    from metal.bitmagic import NULL_TAG
    _HAS_NANBOX = True
except ImportError:
    _HAS_NANBOX = False
    NULL_TAG = 0x7FF8000000000001

try:
    from metal.bitmagic import (
        nanbox_batch_pack_int, nanbox_batch_pack_float)
    _HAS_NANBOX_BATCH = True
except ImportError:
    _HAS_NANBOX_BATCH = False


class VectorBatch:
    """向量化批处理容器。存储多列 DataVector，行数一致。"""

    def __init__(self, columns: Dict[str, DataVector],
                 _row_count: int = 0,
                 _column_order: Optional[List[str]] = None
                 ) -> None:
        self.columns = columns
        self._row_count = _row_count
        self._column_order = _column_order or list(columns.keys())
        if columns:
            lengths = [len(v) for v in columns.values()]
            if lengths and any(l != lengths[0] for l in lengths):
                raise ExecutionError("VectorBatch 列长度不一致")

    @property
    def row_count(self) -> int:
        if self.columns:
            return len(next(iter(self.columns.values())))
        return self._row_count

    @property
    def column_names(self) -> List[str]:
        return list(self._column_order)

    def get_column(self, name: str) -> DataVector:
        if name not in self.columns:
            raise ColumnNotFoundError(name)
        return self.columns[name]

    def add_column(self, name: str,
                   vec: DataVector) -> None:
        self.columns[name] = vec
        if name not in self._column_order:
            self._column_order.append(name)

    def filter_by_bitmap(self,
                         mask: BitmapLike) -> 'VectorBatch':
        indices = mask.to_indices()
        new_cols = {
            n: self.columns[n].filter_by_indices(indices)
            for n in self._column_order}
        return VectorBatch(
            new_cols, _row_count=len(indices),
            _column_order=list(self._column_order))

    def slice(self, start: int,
              end: int) -> 'VectorBatch':
        actual_end = min(end, self.row_count)
        indices = list(range(start, actual_end))
        new_cols = {
            n: self.columns[n].filter_by_indices(indices)
            for n in self._column_order}
        return VectorBatch(
            new_cols, _row_count=len(indices),
            _column_order=list(self._column_order))

    def reorder_by_indices(self,
                           indices: List[int]) -> 'VectorBatch':
        new_cols = {
            n: self.columns[n].filter_by_indices(indices)
            for n in self._column_order}
        return VectorBatch(
            new_cols,
            _column_order=list(self._column_order))

    def to_rows(self) -> List[list]:
        rc = self.row_count
        return [[self.columns[n].get(i)
                 for n in self._column_order]
                for i in range(rc)]

    @staticmethod
    def from_rows(rows: List[list],
                  col_names: List[str],
                  col_types: List[DataType]) -> 'VectorBatch':
        """[P01] 列优先批量构建。
        优化：TypedVector 用初始化列表一次性构建，避免逐值 append。
        [D03] 不使用 Arena（Arena 适合算子级生命周期，此处是临时操作）。"""
        if not rows:
            return VectorBatch.empty(col_names, col_types)

        n = len(rows)
        num_cols = len(col_names)
        cols: Dict[str, DataVector] = {}

        for ci in range(num_cols):
            if ci >= len(col_types):
                break
            cname = col_names[ci]
            ctype = col_types[ci]

            # [P01] 列优先提取
            col_values = [
                rows[ri][ci] if ci < len(rows[ri]) else None
                for ri in range(n)]
            null_indices = [
                ri for ri, v in enumerate(col_values)
                if v is None]
            nulls = Bitmap(n)
            for ni in null_indices:
                nulls.set_bit(ni)

            code = DTYPE_TO_ARRAY_CODE.get(ctype)

            # NaN-Boxing 路径（INT/FLOAT/DOUBLE/DATE）
            if (code is not None and _HAS_NANBOX_BATCH
                    and ctype in _NANBOX_TYPES):
                is_float = ctype in (
                    DataType.FLOAT, DataType.DOUBLE)
                null_set = set(null_indices)
                if is_float:
                    packed = nanbox_batch_pack_float(
                        col_values, null_set, n)
                else:
                    packed = nanbox_batch_pack_int(
                        col_values, null_set, n)
                lazy_data = _unpack_to_typed(
                    ctype, packed, n)
                cols[cname] = DataVector(
                    dtype=ctype, data=lazy_data,
                    nulls=nulls, _length=n, _packed=packed)
                continue

            # BIGINT/TIMESTAMP + 其他数值
            if code is not None:
                # [P01] 一次性构建 TypedVector
                typed_values = [
                    0 if v is None else v
                    for v in col_values]
                try:
                    data: Any = TypedVector(code, typed_values)
                except Exception:
                    data = TypedVector(code)
                    for v in typed_values:
                        try:
                            data.append(v)
                        except Exception:
                            data.append(0)
                cols[cname] = DataVector(
                    dtype=ctype, data=data,
                    nulls=nulls, _length=n)

            elif ctype in (DataType.VARCHAR, DataType.TEXT):
                str_data = [
                    '' if v is None else str(v)
                    for v in col_values]
                cols[cname] = DataVector(
                    dtype=ctype, data=str_data,
                    nulls=nulls, _length=n)

            elif ctype == DataType.BOOLEAN:
                bool_data = Bitmap(n)
                for ri in range(n):
                    if (col_values[ri] is not None
                            and col_values[ri]):
                        bool_data.set_bit(ri)
                cols[cname] = DataVector(
                    dtype=ctype, data=bool_data,
                    nulls=nulls, _length=n)
            else:
                typed_values = [
                    0 if v is None else v
                    for v in col_values]
                cols[cname] = DataVector(
                    dtype=ctype,
                    data=TypedVector('q', typed_values),
                    nulls=nulls, _length=n)

        return VectorBatch(
            cols, _column_order=list(col_names))

    @staticmethod
    def empty(col_names: List[str],
              col_types: List[DataType]) -> 'VectorBatch':
        """构建零行 VectorBatch。"""
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
            cols[cname] = DataVector(
                dtype=ctype, data=data,
                nulls=Bitmap(0), _length=0)
        return VectorBatch(
            cols, _row_count=0,
            _column_order=list(col_names))

    @staticmethod
    def single_row_no_columns() -> 'VectorBatch':
        """单行零列 batch（SELECT 无 FROM 时使用）。"""
        return VectorBatch(columns={}, _row_count=1)

    @staticmethod
    def merge(batches: 'list[VectorBatch]') -> 'VectorBatch':
        """合并多个 batch。"""
        if not batches:
            raise ExecutionError("空 batch 列表")
        col_order = batches[0].column_names
        merged = {
            name: DataVector.concat(
                [b.columns[name] for b in batches])
            for name in col_order}
        return VectorBatch(
            columns=merged,
            _row_count=sum(
                b.row_count for b in batches),
            _column_order=list(col_order))
