from __future__ import annotations
"""DualScan + GenerateSeries scan operators."""
from typing import Any, List, Optional, Tuple
from executor.core.batch import VectorBatch
from executor.core.operator import Operator
from executor.core.vector import DataVector
from metal.bitmap import Bitmap
from metal.typed_vector import TypedVector
from storage.types import DataType


class DualScan(Operator):
    """Produces a single row with no columns."""
    def __init__(self) -> None:
        super().__init__()
        self._emitted = False
    def output_schema(self) -> List[Tuple[str, DataType]]: return []
    def open(self) -> None: self._emitted = False
    def close(self) -> None: pass
    def next_batch(self) -> Optional[VectorBatch]:
        if self._emitted: return None
        self._emitted = True
        return VectorBatch.single_row_no_columns()


class GenerateSeriesOperator(Operator):
    """GENERATE_SERIES(start, stop[, step]) as a table function."""
    def __init__(self, start: int, stop: int, step: int = 1, col_name: str = 'generate_series') -> None:
        super().__init__()
        self._start = start; self._stop = stop; self._step = step
        self._col_name = col_name; self._emitted = False

    def output_schema(self) -> List[Tuple[str, DataType]]:
        return [(self._col_name, DataType.BIGINT)]

    def open(self) -> None: self._emitted = False
    def close(self) -> None: pass

    def next_batch(self) -> Optional[VectorBatch]:
        if self._emitted: return None
        self._emitted = True
        values = list(range(self._start, self._stop + (1 if self._step > 0 else -1), self._step))
        n = len(values)
        data = TypedVector('q', values)
        nulls = Bitmap(n)
        vec = DataVector(dtype=DataType.BIGINT, data=data, nulls=nulls, _length=n)
        return VectorBatch(columns={self._col_name: vec},
                           _column_order=[self._col_name], _row_count=n)
