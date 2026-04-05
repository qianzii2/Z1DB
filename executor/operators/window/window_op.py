from __future__ import annotations
"""Window function operator — adaptive frame computation."""
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

# Adaptive imports
try:
    from structures.segment_tree import SegmentTree, MinSegmentTree, MaxSegmentTree
    _HAS_SEGTREE = True
except ImportError:
    _HAS_SEGTREE = False

try:
    from structures.sparse_table import SparseTableMin, SparseTableMax
    _HAS_SPARSE = True
except ImportError:
    _HAS_SPARSE = False

try:
    from structures.fenwick_tree import FenwickTree
    _HAS_FENWICK = True
except ImportError:
    _HAS_FENWICK = False


class WindowOperator(Operator):
    def __init__(self, child: Operator,
                 window_specs: List[Tuple[str, WindowCall]]) -> None:
        super().__init__()
        self.child = child; self.children = [child]
        self._specs = window_specs
        self._evaluator = ExpressionEvaluator()
        self._result: Optional[VectorBatch] = None
        self._emitted = False

    def output_schema(self) -> List[Tuple[str, DataType]]:
        base = self.child.output_schema()
        extra = []
        for name, wc in self._specs:
            extra.append((name, self._infer_window_type(wc)))
        return base + extra

    def _infer_window_type(self, wc: WindowCall) -> DataType:
        fn = wc.func
        if isinstance(fn, FunctionCall):
            u = fn.name.upper()
            if u in ('ROW_NUMBER','RANK','DENSE_RANK','NTILE'): return DataType.BIGINT
            if u in ('PERCENT_RANK','CUME_DIST'): return DataType.DOUBLE
        if isinstance(fn, AggregateCall):
            if fn.name.upper() == 'COUNT': return DataType.BIGINT
            if fn.name.upper() == 'AVG': return DataType.DOUBLE
        return DataType.UNKNOWN

    def open(self) -> None:
        self.child.open()
        batches = []
        while True:
            b = self.child.next_batch()
            if b is None: break
            batches.append(b)
        self.child.close()
        if not batches:
            schema = self.output_schema()
            self._result = VectorBatch.empty([n for n,_ in schema],[t for _,t in schema])
            self._emitted = False; return
        merged = VectorBatch.merge(batches)
        n = merged.row_count
        for temp_name, wc in self._specs:
            values = self._compute_window(wc, merged, n)
            dt = self._detect_type(values)
            merged.add_column(temp_name, self._list_to_vec(values, dt, n))
        self._result = merged; self._emitted = False

    def next_batch(self) -> Optional[VectorBatch]:
        if self._emitted: return None
        self._emitted = True; return self._result

    def close(self) -> None: pass

    def _compute_window(self, wc: WindowCall, batch: VectorBatch, n: int) -> list:
        part_vals = [self._evaluator.evaluate(e, batch).to_python_list() for e in wc.partition_by]
        sort_exprs = [(self._evaluator.evaluate(sk.expr, batch).to_python_list(), sk.direction, sk.nulls)
                      for sk in wc.order_by]
        partitions: Dict[tuple, List[int]] = {}
        part_order: List[tuple] = []
        for i in range(n):
            key = tuple(pv[i] for pv in part_vals) if part_vals else ()
            if key not in partitions:
                partitions[key] = []; part_order.append(key)
            partitions[key].append(i)
        for key in part_order:
            if sort_exprs:
                partitions[key].sort(key=functools.cmp_to_key(
                    lambda i, j: self._cmp_rows(i, j, sort_exprs)))

        results = [None] * n
        fn = wc.func
        fn_name = (fn.name if isinstance(fn, (FunctionCall, AggregateCall)) else '').upper()

        for key in part_order:
            indices = partitions[key]; psize = len(indices)
            if fn_name == 'ROW_NUMBER':
                for rank, idx in enumerate(indices, 1): results[idx] = rank
            elif fn_name == 'RANK':
                self._compute_rank(indices, sort_exprs, results, False)
            elif fn_name == 'DENSE_RANK':
                self._compute_rank(indices, sort_exprs, results, True)
            elif fn_name == 'NTILE':
                ntiles = int(fn.args[0].value) if fn.args else 1
                for pos, idx in enumerate(indices):
                    results[idx] = (pos * ntiles) // psize + 1
            elif fn_name == 'PERCENT_RANK':
                self._compute_rank(indices, sort_exprs, results, False)
                for idx in indices:
                    results[idx] = (results[idx] - 1) / max(psize - 1, 1)
            elif fn_name == 'CUME_DIST':
                self._compute_rank(indices, sort_exprs, results, False)
                for pos, idx in enumerate(indices):
                    count = pos + 1
                    while count < psize and self._rows_equal(indices[count], idx, sort_exprs):
                        count += 1
                    results[idx] = count / psize
            elif fn_name in ('LAG', 'LEAD'):
                offset = 1; default = None
                if isinstance(fn, FunctionCall) and len(fn.args) >= 2:
                    offset = self._evaluator.evaluate(fn.args[1], batch).get(0) or 1
                    offset = int(offset)
                if isinstance(fn, FunctionCall) and len(fn.args) >= 3:
                    default = self._evaluator.evaluate(fn.args[2], batch).get(0)
                arg_vals = self._evaluator.evaluate(fn.args[0], batch).to_python_list() if fn.args else None
                for pos, idx in enumerate(indices):
                    src = pos - offset if fn_name == 'LAG' else pos + offset
                    if 0 <= src < psize and arg_vals:
                        results[idx] = arg_vals[indices[src]]
                    else:
                        results[idx] = default
            elif fn_name in ('FIRST_VALUE', 'LAST_VALUE', 'NTH_VALUE'):
                arg_vals = self._evaluator.evaluate(fn.args[0], batch).to_python_list() if fn.args else [None]*n
                for pos, idx in enumerate(indices):
                    if fn_name == 'FIRST_VALUE': results[idx] = arg_vals[indices[0]]
                    elif fn_name == 'LAST_VALUE': results[idx] = arg_vals[idx]
                    elif fn_name == 'NTH_VALUE':
                        nth = int(fn.args[1].value) if len(fn.args) > 1 else 1
                        results[idx] = arg_vals[indices[nth-1]] if nth <= psize else None
            elif isinstance(fn, AggregateCall):
                self._compute_agg_window(fn, wc.frame, indices, batch, results)
        return results

    def _compute_rank(self, indices, sort_exprs, results, dense):
        if not indices: return
        rank = 1; dense_rank = 1; results[indices[0]] = 1
        for pos in range(1, len(indices)):
            if not self._rows_equal(indices[pos], indices[pos-1], sort_exprs):
                if dense: dense_rank += 1; rank = dense_rank
                else: rank = pos + 1
            results[indices[pos]] = rank

    def _rows_equal(self, i, j, sort_exprs):
        for vals, _, _ in sort_exprs:
            if vals[i] != vals[j]: return False
        return True

    def _compute_agg_window(self, fn: AggregateCall, frame: Optional[WindowFrame],
                            indices: list, batch: VectorBatch, results: list) -> None:
        name = fn.name.upper()
        if fn.args and not isinstance(fn.args[0], StarExpr):
            arg_vals = self._evaluator.evaluate(fn.args[0], batch).to_python_list()
        else:
            arg_vals = None
        psize = len(indices)

        # Collect non-null values for this partition
        partition_vals = []
        for pos in range(psize):
            if arg_vals is not None:
                v = arg_vals[indices[pos]]
                partition_vals.append(v if v is not None else 0)
            else:
                partition_vals.append(1)

        # ═══ Adaptive strategy selection ═══
        use_prefix = frame is None  # UNBOUNDED PRECEDING TO CURRENT ROW
        frame_start_type = frame.start.type if frame and frame.start else 'UNBOUNDED_PRECEDING'
        frame_end_type = frame.end.type if frame and frame.end else 'CURRENT_ROW'

        if use_prefix or (frame_start_type == 'UNBOUNDED_PRECEDING' and frame_end_type == 'CURRENT_ROW'):
            # O(n) prefix sum
            self._agg_prefix(name, partition_vals, indices, results, arg_vals)
            return

        # Fixed frame size?
        fixed_start = frame.start.offset if frame and frame.start and frame.start.type == 'N_PRECEDING' else None
        fixed_end = frame.end.offset if frame and frame.end and frame.end.type == 'N_FOLLOWING' else None

        if fixed_start is not None or fixed_end is not None:
            frame_size = (fixed_start or 0) + (fixed_end or 0) + 1
            if frame_size <= 8:
                # Brute force — small frame
                self._agg_brute(name, partition_vals, indices, results, arg_vals, frame, psize)
                return
            if name in ('MIN', 'MAX') and _HAS_SPARSE and psize > 16:
                self._agg_sparse_table(name, partition_vals, indices, results, frame, psize)
                return
            if name in ('SUM', 'AVG', 'COUNT') and _HAS_SEGTREE and psize > 16:
                self._agg_segment_tree(name, partition_vals, indices, results, frame, psize)
                return

        # Fallback: brute force
        self._agg_brute(name, partition_vals, indices, results, arg_vals, frame, psize)

    def _agg_prefix(self, name, vals, indices, results, arg_vals):
        """O(n) prefix approach for UNBOUNDED PRECEDING TO CURRENT ROW."""
        if name == 'COUNT':
            cnt = 0
            for pos, idx in enumerate(indices):
                v = arg_vals[idx] if arg_vals else 1
                if v is not None: cnt += 1
                results[idx] = cnt
        elif name == 'SUM':
            s = 0
            for pos, idx in enumerate(indices):
                v = vals[pos]
                if arg_vals is None or arg_vals[idx] is not None: s += v
                results[idx] = s
        elif name == 'AVG':
            s = 0; cnt = 0
            for pos, idx in enumerate(indices):
                v = vals[pos]
                if arg_vals is None or arg_vals[idx] is not None:
                    s += v; cnt += 1
                results[idx] = s / cnt if cnt > 0 else None
        elif name == 'MIN':
            cur_min = None
            for pos, idx in enumerate(indices):
                v = vals[pos]
                if arg_vals is None or arg_vals[idx] is not None:
                    if cur_min is None or v < cur_min: cur_min = v
                results[idx] = cur_min
        elif name == 'MAX':
            cur_max = None
            for pos, idx in enumerate(indices):
                v = vals[pos]
                if arg_vals is None or arg_vals[idx] is not None:
                    if cur_max is None or v > cur_max: cur_max = v
                results[idx] = cur_max

    def _agg_segment_tree(self, name, vals, indices, results, frame, psize):
        """O(n log n) via Segment Tree for fixed-frame SUM/COUNT."""
        non_null = [v for v in vals]
        st = SegmentTree(non_null)
        for pos in range(psize):
            start, end = self._frame_bounds(frame, pos, psize)
            s = st.query(start, end)
            if name == 'SUM': results[indices[pos]] = s
            elif name == 'COUNT': results[indices[pos]] = end - start + 1
            elif name == 'AVG':
                cnt = end - start + 1
                results[indices[pos]] = s / cnt if cnt > 0 else None

    def _agg_sparse_table(self, name, vals, indices, results, frame, psize):
        """O(1) per query via Sparse Table for fixed-frame MIN/MAX."""
        if name == 'MIN':
            st = SparseTableMin(vals)
        else:
            st = SparseTableMax(vals)
        for pos in range(psize):
            start, end = self._frame_bounds(frame, pos, psize)
            results[indices[pos]] = st.query(start, end)

    def _agg_brute(self, name, vals, indices, results, arg_vals, frame, psize):
        """Brute force frame computation."""
        for pos in range(psize):
            start, end = self._frame_bounds(frame, pos, psize)
            window_vals = []
            for wp in range(start, end + 1):
                if arg_vals is not None:
                    v = arg_vals[indices[wp]]
                    if v is not None: window_vals.append(v)
                else:
                    window_vals.append(1)
            if name == 'COUNT': results[indices[pos]] = len(window_vals)
            elif name == 'SUM': results[indices[pos]] = sum(window_vals) if window_vals else None
            elif name == 'AVG': results[indices[pos]] = sum(window_vals)/len(window_vals) if window_vals else None
            elif name == 'MIN': results[indices[pos]] = min(window_vals) if window_vals else None
            elif name == 'MAX': results[indices[pos]] = max(window_vals) if window_vals else None

    def _frame_bounds(self, frame, pos, psize):
        if frame is None: return 0, pos
        start = 0; end = pos
        if frame.start:
            if frame.start.type == 'UNBOUNDED_PRECEDING': start = 0
            elif frame.start.type == 'CURRENT_ROW': start = pos
            elif frame.start.type == 'N_PRECEDING': start = max(0, pos-(frame.start.offset or 0))
            elif frame.start.type == 'N_FOLLOWING': start = min(psize-1, pos+(frame.start.offset or 0))
        if frame.end:
            if frame.end.type == 'UNBOUNDED_FOLLOWING': end = psize-1
            elif frame.end.type == 'CURRENT_ROW': end = pos
            elif frame.end.type == 'N_PRECEDING': end = max(0, pos-(frame.end.offset or 0))
            elif frame.end.type == 'N_FOLLOWING': end = min(psize-1, pos+(frame.end.offset or 0))
        return start, end

    def _cmp_rows(self, i, j, sort_exprs):
        for vals, direction, nulls_pos in sort_exprs:
            np = nulls_pos or ('NULLS_LAST' if direction == 'ASC' else 'NULLS_FIRST')
            a, b = vals[i], vals[j]
            if a is None and b is None: continue
            if a is None: return 1 if np == 'NULLS_LAST' else -1
            if b is None: return -1 if np == 'NULLS_LAST' else 1
            if a < b: cmp = -1
            elif a > b: cmp = 1
            else: continue
            return -cmp if direction == 'DESC' else cmp
        return 0

    def _detect_type(self, values):
        for v in values:
            if v is None: continue
            if isinstance(v, bool): return DataType.BOOLEAN
            if isinstance(v, int): return DataType.BIGINT
            if isinstance(v, float): return DataType.DOUBLE
            if isinstance(v, str): return DataType.VARCHAR
        return DataType.BIGINT

    def _list_to_vec(self, values, dtype, n):
        if dtype == DataType.UNKNOWN: dtype = DataType.BIGINT
        code = DTYPE_TO_ARRAY_CODE.get(dtype); nulls = Bitmap(n)
        if dtype in (DataType.VARCHAR, DataType.TEXT):
            data: Any = []
            for i in range(n):
                if values[i] is None: nulls.set_bit(i); data.append('')
                else: data.append(str(values[i]))
        elif dtype == DataType.BOOLEAN:
            data = Bitmap(n)
            for i in range(n):
                if values[i] is None: nulls.set_bit(i)
                elif values[i]: data.set_bit(i)
        elif code:
            data = TypedVector(code)
            for i in range(n):
                if values[i] is None: nulls.set_bit(i); data.append(0)
                else: data.append(values[i])
        else:
            data = TypedVector('q')
            for i in range(n):
                if values[i] is None: nulls.set_bit(i); data.append(0)
                else: data.append(int(values[i]))
        return DataVector(dtype=dtype, data=data, nulls=nulls, _length=n)
