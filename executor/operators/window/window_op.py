from __future__ import annotations
"""Window function operator."""
import functools
from typing import Any, Dict, List, Optional, Tuple
from executor.core.batch import VectorBatch
from executor.core.operator import Operator
from executor.core.vector import DataVector
from executor.expression.evaluator import ExpressionEvaluator
from metal.bitmap import Bitmap
from metal.typed_vector import TypedVector
from parser.ast import AggregateCall, FunctionCall, SortKey, StarExpr, WindowCall, WindowFrame
from storage.types import DTYPE_TO_ARRAY_CODE, DataType


class WindowOperator(Operator):
    """Computes window functions over partitions."""

    def __init__(self, child: Operator,
                 window_specs: List[Tuple[str, WindowCall]]) -> None:
        super().__init__()
        self.child = child
        self.children = [child]
        self._specs = window_specs  # [(temp_col_name, WindowCall)]
        self._evaluator = ExpressionEvaluator()
        self._result: Optional[VectorBatch] = None
        self._emitted = False

    def output_schema(self) -> List[Tuple[str, DataType]]:
        base = self.child.output_schema()
        extra = []
        for name, wc in self._specs:
            dt = self._infer_window_type(wc)
            extra.append((name, dt))
        return base + extra

    def _infer_window_type(self, wc: WindowCall) -> DataType:
        fn = wc.func
        if isinstance(fn, FunctionCall):
            upper = fn.name.upper()
            if upper in ('ROW_NUMBER', 'RANK', 'DENSE_RANK', 'NTILE'):
                return DataType.BIGINT
            if upper in ('PERCENT_RANK', 'CUME_DIST'):
                return DataType.DOUBLE
            if upper in ('LAG', 'LEAD', 'FIRST_VALUE', 'LAST_VALUE', 'NTH_VALUE'):
                return DataType.UNKNOWN  # determined at runtime
        if isinstance(fn, AggregateCall):
            upper = fn.name.upper()
            if upper == 'COUNT':
                return DataType.BIGINT
            if upper == 'AVG':
                return DataType.DOUBLE
        return DataType.UNKNOWN

    def open(self) -> None:
        self.child.open()
        batches = []
        while True:
            b = self.child.next_batch()
            if b is None:
                break
            batches.append(b)
        self.child.close()

        if not batches:
            schema = self.output_schema()
            self._result = VectorBatch.empty([n for n, _ in schema], [t for _, t in schema])
            self._emitted = False
            return

        merged = VectorBatch.merge(batches)
        n = merged.row_count

        for temp_name, wc in self._specs:
            values = self._compute_window(wc, merged, n)
            dt = self._detect_type(values)
            vec = self._list_to_vec(values, dt, n)
            merged.add_column(temp_name, vec)

        self._result = merged
        self._emitted = False

    def next_batch(self) -> Optional[VectorBatch]:
        if self._emitted:
            return None
        self._emitted = True
        return self._result

    def close(self) -> None:
        pass

    def _compute_window(self, wc: WindowCall, batch: VectorBatch, n: int) -> list:
        # Build partition keys
        part_vals = [self._evaluator.evaluate(e, batch).to_python_list() for e in wc.partition_by]
        # Build sort keys within partition
        sort_exprs = [(self._evaluator.evaluate(sk.expr, batch).to_python_list(), sk.direction, sk.nulls)
                      for sk in wc.order_by]

        # Assign partition ids
        partitions: Dict[tuple, List[int]] = {}
        part_order: List[tuple] = []
        for i in range(n):
            key = tuple(pv[i] for pv in part_vals) if part_vals else ()
            if key not in partitions:
                partitions[key] = []
                part_order.append(key)
            partitions[key].append(i)

        # Sort within each partition
        for key in part_order:
            indices = partitions[key]
            if sort_exprs:
                indices.sort(key=functools.cmp_to_key(
                    lambda i, j: self._cmp_rows(i, j, sort_exprs)))
                partitions[key] = indices

        results = [None] * n
        fn = wc.func
        fn_name = (fn.name if isinstance(fn, FunctionCall) else fn.name if isinstance(fn, AggregateCall) else '').upper()

        for key in part_order:
            indices = partitions[key]
            psize = len(indices)

            if fn_name == 'ROW_NUMBER':
                for rank, idx in enumerate(indices, 1):
                    results[idx] = rank
            elif fn_name == 'RANK':
                self._compute_rank(indices, sort_exprs, results, dense=False)
            elif fn_name == 'DENSE_RANK':
                self._compute_rank(indices, sort_exprs, results, dense=True)
            elif fn_name == 'NTILE':
                ntiles = int(fn.args[0].value) if fn.args else 1
                for pos, idx in enumerate(indices):
                    results[idx] = (pos * ntiles) // psize + 1
            elif fn_name == 'PERCENT_RANK':
                self._compute_rank(indices, sort_exprs, results, dense=False)
                for idx in indices:
                    results[idx] = (results[idx] - 1) / max(psize - 1, 1)
            elif fn_name == 'CUME_DIST':
                self._compute_rank(indices, sort_exprs, results, dense=False)
                # cume_dist = count of rows <= current / total
                for pos, idx in enumerate(indices):
                    # Find how many share same rank or lower
                    count = pos + 1
                    while count < psize and self._rows_equal(indices[count], idx, sort_exprs):
                        count += 1
                    results[idx] = count / psize
            elif fn_name in ('LAG', 'LEAD'):
                offset = 1
                default = None
                if isinstance(fn, FunctionCall) and len(fn.args) >= 2:
                    offset = self._evaluator.evaluate(fn.args[1], batch).get(0)
                    if offset is None:
                        offset = 1
                    offset = int(offset)
                if isinstance(fn, FunctionCall) and len(fn.args) >= 3:
                    default = self._evaluator.evaluate(fn.args[2], batch).get(0)
                arg_vals = None
                if isinstance(fn, FunctionCall) and fn.args:
                    arg_vals = self._evaluator.evaluate(fn.args[0], batch).to_python_list()
                for pos, idx in enumerate(indices):
                    src_pos = pos - offset if fn_name == 'LAG' else pos + offset
                    if 0 <= src_pos < psize and arg_vals is not None:
                        results[idx] = arg_vals[indices[src_pos]]
                    else:
                        results[idx] = default
            elif fn_name in ('FIRST_VALUE', 'LAST_VALUE', 'NTH_VALUE'):
                if isinstance(fn, FunctionCall) and fn.args:
                    arg_vals = self._evaluator.evaluate(fn.args[0], batch).to_python_list()
                else:
                    arg_vals = [None] * n
                for pos, idx in enumerate(indices):
                    if fn_name == 'FIRST_VALUE':
                        results[idx] = arg_vals[indices[0]]
                    elif fn_name == 'LAST_VALUE':
                        # Default frame: UNBOUNDED PRECEDING to CURRENT ROW
                        results[idx] = arg_vals[idx]
                    elif fn_name == 'NTH_VALUE':
                        nth = int(fn.args[1].value) if len(fn.args) > 1 else 1
                        results[idx] = arg_vals[indices[nth-1]] if nth <= psize else None
            elif isinstance(fn, AggregateCall):
                self._compute_agg_window(fn, wc.frame, indices, batch, results)

        return results

    def _compute_rank(self, indices: list, sort_exprs: list,
                      results: list, dense: bool) -> None:
        if not indices:
            return
        rank = 1
        results[indices[0]] = 1
        dense_rank = 1
        for pos in range(1, len(indices)):
            if not self._rows_equal(indices[pos], indices[pos-1], sort_exprs):
                if dense:
                    dense_rank += 1
                    rank = dense_rank
                else:
                    rank = pos + 1
            results[indices[pos]] = rank

    def _rows_equal(self, i: int, j: int, sort_exprs: list) -> bool:
        for vals, _, _ in sort_exprs:
            if vals[i] != vals[j]:
                return False
        return True

    def _compute_agg_window(self, fn: AggregateCall, frame: Optional[WindowFrame],
                            indices: list, batch: VectorBatch, results: list) -> None:
        name = fn.name.upper()
        if fn.args and not isinstance(fn.args[0], StarExpr):
            arg_vals = self._evaluator.evaluate(fn.args[0], batch).to_python_list()
        else:
            arg_vals = None

        psize = len(indices)
        for pos, idx in enumerate(indices):
            start, end = self._frame_bounds(frame, pos, psize)
            window_vals = []
            for wp in range(start, end + 1):
                if arg_vals is not None:
                    v = arg_vals[indices[wp]]
                    if v is not None:
                        window_vals.append(v)
                else:
                    window_vals.append(1)  # COUNT(*)

            if name == 'COUNT':
                results[idx] = len(window_vals)
            elif name == 'SUM':
                results[idx] = sum(window_vals) if window_vals else None
            elif name == 'AVG':
                results[idx] = sum(window_vals) / len(window_vals) if window_vals else None
            elif name == 'MIN':
                results[idx] = min(window_vals) if window_vals else None
            elif name == 'MAX':
                results[idx] = max(window_vals) if window_vals else None

    def _frame_bounds(self, frame: Optional[WindowFrame],
                      pos: int, psize: int) -> Tuple[int, int]:
        if frame is None:
            return 0, pos  # default: UNBOUNDED PRECEDING to CURRENT ROW
        start = 0
        end = pos
        if frame.start:
            if frame.start.type == 'UNBOUNDED_PRECEDING':
                start = 0
            elif frame.start.type == 'CURRENT_ROW':
                start = pos
            elif frame.start.type == 'N_PRECEDING':
                start = max(0, pos - (frame.start.offset or 0))
            elif frame.start.type == 'N_FOLLOWING':
                start = min(psize - 1, pos + (frame.start.offset or 0))
        if frame.end:
            if frame.end.type == 'UNBOUNDED_FOLLOWING':
                end = psize - 1
            elif frame.end.type == 'CURRENT_ROW':
                end = pos
            elif frame.end.type == 'N_PRECEDING':
                end = max(0, pos - (frame.end.offset or 0))
            elif frame.end.type == 'N_FOLLOWING':
                end = min(psize - 1, pos + (frame.end.offset or 0))
        return start, end

    def _cmp_rows(self, i: int, j: int, sort_exprs: list) -> int:
        for vals, direction, nulls_pos in sort_exprs:
            np = nulls_pos or ('NULLS_LAST' if direction == 'ASC' else 'NULLS_FIRST')
            a, b = vals[i], vals[j]
            an, bn = a is None, b is None
            if an and bn:
                continue
            if an:
                return 1 if np == 'NULLS_LAST' else -1
            if bn:
                return -1 if np == 'NULLS_LAST' else 1
            if a < b:
                cmp = -1
            elif a > b:
                cmp = 1
            else:
                continue
            return -cmp if direction == 'DESC' else cmp
        return 0

    def _detect_type(self, values: list) -> DataType:
        for v in values:
            if v is None:
                continue
            if isinstance(v, bool):
                return DataType.BOOLEAN
            if isinstance(v, int):
                return DataType.BIGINT
            if isinstance(v, float):
                return DataType.DOUBLE
            if isinstance(v, str):
                return DataType.VARCHAR
        return DataType.BIGINT

    def _list_to_vec(self, values: list, dtype: DataType, n: int) -> DataVector:
        if dtype == DataType.UNKNOWN:
            dtype = DataType.BIGINT
        code = DTYPE_TO_ARRAY_CODE.get(dtype)
        nulls = Bitmap(n)
        if dtype in (DataType.VARCHAR, DataType.TEXT):
            data: Any = []
            for i in range(n):
                if values[i] is None:
                    nulls.set_bit(i)
                    data.append('')
                else:
                    data.append(str(values[i]))
        elif dtype == DataType.BOOLEAN:
            data = Bitmap(n)
            for i in range(n):
                if values[i] is None:
                    nulls.set_bit(i)
                elif values[i]:
                    data.set_bit(i)
        elif code:
            data = TypedVector(code)
            for i in range(n):
                if values[i] is None:
                    nulls.set_bit(i)
                    data.append(0)
                else:
                    data.append(values[i])
        else:
            data = TypedVector('q')
            for i in range(n):
                if values[i] is None:
                    nulls.set_bit(i)
                    data.append(0)
                else:
                    data.append(int(values[i]))
        return DataVector(dtype=dtype, data=data, nulls=nulls, _length=n)
