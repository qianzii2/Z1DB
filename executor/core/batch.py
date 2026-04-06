from __future__ import annotations
"""VectorBatch — Arena 加速的批量构建。
[集成] Arena 用于 from_rows 的临时缓冲区分配。
批处理完成后 Arena 一次性释放。"""
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
    from metal.arena import Arena
    _HAS_ARENA = True
except ImportError:
    _HAS_ARENA = False

try:
    from metal.slab import SlabAllocator
    _HAS_SLAB = True
except ImportError:
    _HAS_SLAB = False

# Slab：固定大小 VectorBatch 头部对象池
_BATCH_SLAB: Optional[SlabAllocator] = None
_SLAB_SLOT_SIZE = 128  # 每个 slot 存 batch 元数据


def _get_batch_slab() -> Optional[SlabAllocator]:
    global _BATCH_SLAB
    if _HAS_SLAB and _BATCH_SLAB is None:
        try:
            _BATCH_SLAB = SlabAllocator(_SLAB_SLOT_SIZE, slab_capacity=1024)
        except Exception:
            pass
    return _BATCH_SLAB


class VectorBatch:
    def __init__(self, columns: Dict[str, DataVector],
                 _row_count: int = 0,
                 _column_order: Optional[List[str]] = None) -> None:
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

    def add_column(self, name: str, vec: DataVector) -> None:
        self.columns[name] = vec
        if name not in self._column_order:
            self._column_order.append(name)

    def filter_by_bitmap(self, mask: BitmapLike) -> 'VectorBatch':
        indices = mask.to_indices()
        new_cols = {n: self.columns[n].filter_by_indices(indices)
                    for n in self._column_order}
        return VectorBatch(new_cols, _row_count=len(indices),
                           _column_order=list(self._column_order))

    def slice(self, start: int, end: int) -> 'VectorBatch':
        actual_end = min(end, self.row_count)
        indices = list(range(start, actual_end))
        new_cols = {n: self.columns[n].filter_by_indices(indices)
                    for n in self._column_order}
        return VectorBatch(new_cols, _row_count=len(indices),
                           _column_order=list(self._column_order))

    def reorder_by_indices(self, indices: List[int]) -> 'VectorBatch':
        new_cols = {n: self.columns[n].filter_by_indices(indices)
                    for n in self._column_order}
        return VectorBatch(new_cols, _column_order=list(self._column_order))

    def to_rows(self) -> List[list]:
        rc = self.row_count
        return [[self.columns[n].get(i) for n in self._column_order]
                for i in range(rc)]

    @staticmethod
    def from_rows(rows: List[list], col_names: List[str],
                  col_types: List[DataType]) -> 'VectorBatch':
        """[Arena] 列优先构建。临时缓冲区用 Arena 分配。"""
        if not rows:
            return VectorBatch.empty(col_names, col_types)

        n = len(rows)
        num_cols = len(col_names)
        cols: Dict[str, DataVector] = {}

        # [Arena] 大批量用 Arena 管理临时内存
        arena = None
        if _HAS_ARENA and n > 1024:
            try:
                arena = Arena()
            except Exception:
                arena = None

        for ci in range(num_cols):
            if ci >= len(col_types):
                break
            cname = col_names[ci]
            ctype = col_types[ci]

            # 列优先提取
            col_values = [rows[ri][ci] if ci < len(rows[ri]) else None
                          for ri in range(n)]
            null_indices = [ri for ri, v in enumerate(col_values) if v is None]
            nulls = Bitmap(n)
            for ni in null_indices:
                nulls.set_bit(ni)

            code = DTYPE_TO_ARRAY_CODE.get(ctype)

            # NaN-Boxing 路径（INT/FLOAT/DOUBLE/DATE）
            if code is not None and _HAS_NANBOX and ctype in _NANBOX_TYPES:
                is_float = ctype in (DataType.FLOAT, DataType.DOUBLE)
                from metal.bitmagic import (
                    nanbox_batch_pack_int, nanbox_batch_pack_float)

                null_set = set(null_indices)
                if is_float:
                    packed = nanbox_batch_pack_float(
                        col_values, null_set, n)
                else:
                    packed = nanbox_batch_pack_int(
                        col_values, null_set, n)

                lazy_data = _unpack_to_typed(ctype, packed, n)
                cols[cname] = DataVector(
                    dtype=ctype, data=lazy_data,
                    nulls=nulls, _length=n, _packed=packed)
                continue

            # BIGINT/TIMESTAMP + 其他数值：直接 TypedVector
            if code is not None:
                typed_values = [0 if v is None else v for v in col_values]
                try:
                    data: Any = TypedVector(code, typed_values)
                except Exception:
                    data = TypedVector(code)
                    for v in typed_values:
                        try:
                            data.append(v)
                        except Exception:
                            data.append(0)
                cols[cname] = DataVector(dtype=ctype, data=data,
                                         nulls=nulls, _length=n)

            elif ctype in (DataType.VARCHAR, DataType.TEXT):
                # [Arena] 字符串列：Arena 存储编码后的字节（未来优化）
                str_data = ['' if v is None else str(v) for v in col_values]
                cols[cname] = DataVector(dtype=ctype, data=str_data,
                                         nulls=nulls, _length=n)

            elif ctype == DataType.BOOLEAN:
                bool_data = Bitmap(n)
                for ri in range(n):
                    if col_values[ri] is not None and col_values[ri]:
                        bool_data.set_bit(ri)
                cols[cname] = DataVector(dtype=ctype, data=bool_data,
                                         nulls=nulls, _length=n)
            else:
                typed_values = [0 if v is None else v for v in col_values]
                cols[cname] = DataVector(
                    dtype=ctype, data=TypedVector('q', typed_values),
                    nulls=nulls, _length=n)

        # Arena 在这里不显式释放——Python GC 会回收
        # 未来优化：Arena 持有所有列数据的引用，batch 销毁时 arena.reset()
        return VectorBatch(cols, _column_order=list(col_names))

    @staticmethod
    def empty(col_names: List[str],
              col_types: List[DataType]) -> 'VectorBatch':
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
            cols[cname] = DataVector(dtype=ctype, data=data,
                                     nulls=Bitmap(0), _length=0)
        return VectorBatch(cols, _row_count=0,
                           _column_order=list(col_names))

    @staticmethod
    def single_row_no_columns() -> 'VectorBatch':
        return VectorBatch(columns={}, _row_count=1)

    @staticmethod
    def merge(batches: 'list[VectorBatch]') -> 'VectorBatch':
        if not batches:
            raise ExecutionError("空 batch 列表")
        col_order = batches[0].column_names
        merged = {name: DataVector.concat([b.columns[name] for b in batches])
                  for name in col_order}
        return VectorBatch(columns=merged,
                           _row_count=sum(b.row_count for b in batches),
                           _column_order=list(col_order))
