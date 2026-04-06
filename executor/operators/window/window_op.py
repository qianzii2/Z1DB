from __future__ import annotations
"""窗口函数算子 — 委托到 ranking.py 和 aggregate.py。"""
import functools
from typing import Any, Dict, List, Optional, Tuple
from executor.core.batch import VectorBatch
from executor.core.operator import Operator
from executor.core.vector import DataVector
from executor.expression.evaluator import ExpressionEvaluator
from metal.bitmap import Bitmap
from parser.ast import AggregateCall, FunctionCall, StarExpr, WindowCall
from storage.types import DTYPE_TO_ARRAY_CODE, DataType
from executor.operators.window.ranking import (
    compute_ranking, compute_navigation, frame_bounds)
from executor.operators.window.aggregate import compute_agg_window

_RANKING_FUNCS = frozenset({
    'ROW_NUMBER', 'RANK', 'DENSE_RANK', 'NTILE',
    'PERCENT_RANK', 'CUME_DIST'})
_NAV_FUNCS = frozenset({
    'LAG', 'LEAD', 'FIRST_VALUE', 'LAST_VALUE', 'NTH_VALUE'})


class WindowOperator(Operator):
    def __init__(self, child: Operator,
                 window_specs: List[Tuple[str, WindowCall]]) -> None:
        super().__init__()
        self.child = child; self.children = [child]
        self._specs = window_specs
        self._evaluator = ExpressionEvaluator()
        self._result: Optional[VectorBatch] = None
        self._emitted = False

    def output_schema(self):
        base = self.child.output_schema()
        return base + [(n, self._infer_type(wc)) for n, wc in self._specs]

    def _infer_type(self, wc):
        fn = wc.func
        if isinstance(fn, FunctionCall):
            u = fn.name.upper()
            if u in ('ROW_NUMBER', 'RANK', 'DENSE_RANK', 'NTILE'): return DataType.BIGINT
            if u in ('PERCENT_RANK', 'CUME_DIST'): return DataType.DOUBLE
        if isinstance(fn, AggregateCall):
            if fn.name.upper() == 'COUNT': return DataType.BIGINT
            if fn.name.upper() in ('AVG', 'MEDIAN', 'PERCENTILE_CONT', 'APPROX_PERCENTILE'):
                return DataType.DOUBLE
        return DataType.UNKNOWN

    def open(self):
        self.child.open()
        batches = []
        while True:
            b = self._ensure_batch(self.child.next_batch())
            if b is None: break
            batches.append(b)
        self.child.close()
        if not batches:
            s = self.output_schema()
            self._result = VectorBatch.empty([n for n, _ in s], [t for _, t in s])
            self._emitted = False; return
        merged = VectorBatch.merge(batches)
        n = merged.row_count
        for temp_name, wc in self._specs:
            values = self._compute_window(wc, merged, n)
            dt = self._detect_type(values)
            merged.add_column(temp_name, self._evaluator._list_to_vec(values, dt, n))
        self._result = merged; self._emitted = False

    def next_batch(self):
        if self._emitted: return None
        self._emitted = True; return self._result

    def close(self): pass

    def _compute_window(self, wc, batch, n):
        pv = [self._evaluator.evaluate(e, batch).to_python_list() for e in wc.partition_by]
        se = [(self._evaluator.evaluate(sk.expr, batch).to_python_list(),
               sk.direction, sk.nulls) for sk in wc.order_by]
        ho = bool(wc.order_by)
        # 分区
        parts: Dict[tuple, List[int]] = {}; po: List[tuple] = []
        for i in range(n):
            k = tuple(p[i] for p in pv) if pv else ()
            if k not in parts: parts[k] = []; po.append(k)
            parts[k].append(i)
        for k in po:
            if se:
                parts[k].sort(key=functools.cmp_to_key(
                    lambda i, j: self._cmp_rows(i, j, se)))

        results = [None] * n
        fn = wc.func
        fn_name = (fn.name if isinstance(fn, (FunctionCall, AggregateCall)) else '').upper()

        for k in po:
            idx = parts[k]
            if fn_name in _RANKING_FUNCS:
                compute_ranking(fn_name, fn, idx, se, batch, results, self._evaluator)
            elif fn_name in _NAV_FUNCS:
                compute_navigation(fn_name, fn, idx, batch, results,
                                    wc.frame, ho, self._evaluator)
            elif isinstance(fn, AggregateCall):
                compute_agg_window(fn, wc.frame, idx, batch, results, ho, self._evaluator)
        return results

    @staticmethod
    def _cmp_rows(i, j, se):
        for v, d, np in se:
            np = np or ('NULLS_LAST' if d == 'ASC' else 'NULLS_FIRST')
            a, b = v[i], v[j]
            if a is None and b is None: continue
            if a is None: return 1 if np == 'NULLS_LAST' else -1
            if b is None: return -1 if np == 'NULLS_LAST' else 1
            if a < b: c = -1
            elif a > b: c = 1
            else: continue
            return -c if d == 'DESC' else c
        return 0

    @staticmethod
    def _detect_type(values):
        for v in values:
            if v is None: continue
            if isinstance(v, bool): return DataType.BOOLEAN
            if isinstance(v, int): return DataType.BIGINT
            if isinstance(v, float): return DataType.DOUBLE
            if isinstance(v, str): return DataType.VARCHAR
        return DataType.BIGINT
