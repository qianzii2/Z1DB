from __future__ import annotations
"""DualScan + GenerateSeries 扫描算子。
R2 修复：GenerateSeries 正确解析 start/stop/step。"""
from typing import Any, List, Optional, Tuple
from executor.core.batch import VectorBatch
from executor.core.operator import Operator
from executor.core.vector import DataVector
from metal.bitmap import Bitmap
from metal.typed_vector import TypedVector
from storage.types import DataType


class DualScan(Operator):
    """产生单行空列。"""
    def __init__(self) -> None:
        super().__init__()
        self._emitted = False
    def output_schema(self): return []
    def open(self): self._emitted = False
    def close(self): pass
    def next_batch(self):
        if self._emitted: return None
        self._emitted = True
        return VectorBatch.single_row_no_columns()


class GenerateSeriesOperator(Operator):
    """GENERATE_SERIES(start, stop[, step]) 表函数。"""
    def __init__(self, start: int, stop: int,
                 step: int = 1,
                 col_name: str = 'generate_series'
                 ) -> None:
        super().__init__()
        self._start = start
        self._stop = stop
        self._step = step if step != 0 else 1
        self._col_name = col_name
        self._emitted = False

    def output_schema(self):
        return [(self._col_name, DataType.BIGINT)]

    def open(self): self._emitted = False
    def close(self): pass

    def next_batch(self):
        if self._emitted: return None
        self._emitted = True
        if self._step > 0:
            values = list(range(
                self._start, self._stop + 1, self._step))
        else:
            values = list(range(
                self._start, self._stop - 1, self._step))
        n = len(values)
        if n == 0:
            return VectorBatch.empty(
                [self._col_name], [DataType.BIGINT])
        data = TypedVector('q', values)
        nulls = Bitmap(n)
        vec = DataVector(
            dtype=DataType.BIGINT, data=data,
            nulls=nulls, _length=n)
        return VectorBatch(
            columns={self._col_name: vec},
            _column_order=[self._col_name],
            _row_count=n)

    @staticmethod
    def parse_args(args_ast: list,
                   evaluator: Any) -> Optional[tuple]:
        """从 AST 参数列表解析 (start, stop, step)。
        供 SimplePlanner._build_generate_series 调用。"""
        from executor.core.batch import VectorBatch
        dummy = VectorBatch.single_row_no_columns()
        try:
            if len(args_ast) < 2:
                return None
            start = int(evaluator.evaluate(
                args_ast[0], dummy).get(0))
            stop = int(evaluator.evaluate(
                args_ast[1], dummy).get(0))
            step = 1
            if len(args_ast) >= 3:
                step = int(evaluator.evaluate(
                    args_ast[2], dummy).get(0))
            return start, stop, step
        except Exception:
            return None
