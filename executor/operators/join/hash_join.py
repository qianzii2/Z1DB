from __future__ import annotations
"""Hash join — group probing for cache efficiency."""
from typing import Any, Dict, List, Optional, Tuple
from executor.core.batch import VectorBatch
from executor.core.operator import Operator
from executor.core.vector import DataVector
from executor.expression.evaluator import ExpressionEvaluator
from metal.hash import murmur3_64
from storage.types import DataType

try:
    from executor.core.lazy_batch import LazyBatch
    _HAS_LAZY = True
except ImportError:
    _HAS_LAZY = False

PROBE_BATCH_SIZE = 16


class HashJoinOperator(Operator):
    """Hash join with group probing for cache efficiency."""

    def __init__(self, left: Operator, right: Operator, join_type: str,
                 on_expr: Any) -> None:
        super().__init__()
        self.left = left; self.right = right; self.children = [left, right]
        self.join_type = join_type
        self._on_expr = on_expr
        self._evaluator = ExpressionEvaluator()
        self._result_rows: List[list] = []
        self._out_names: List[str] = []
        self._out_types: List[DataType] = []
        self._emitted = False

    def output_schema(self) -> List[Tuple[str, DataType]]:
        return self.left.output_schema() + self.right.output_schema()

    def open(self) -> None:
        self.left.open(); self.right.open()
        out_schema = self.output_schema()
        self._out_names = [n for n, _ in out_schema]
        self._out_types = [t for _, t in out_schema]

        # Build phase: materialize right side
        right_rows: List[Dict[str, Any]] = []
        while True:
            b = self.right.next_batch()
            if b is None: break
            if _HAS_LAZY and isinstance(b, LazyBatch):
                b = b.materialize()
            for i in range(b.row_count):
                right_rows.append({n: b.columns[n].get(i) for n in b.column_names})
        self.right.close()

        # Probe phase
        self._result_rows = []
        right_matched: set = set()

        while True:
            lb = self.left.next_batch()
            if lb is None: break
            if _HAS_LAZY and isinstance(lb, LazyBatch):
                lb = lb.materialize()

            for li in range(lb.row_count):
                l_row = {n: lb.columns[n].get(li) for n in lb.column_names}
                found = False
                for ri, r_row in enumerate(right_rows):
                    combined = {**l_row, **r_row}
                    if self._on_expr is None or self._eval_cond(combined, out_schema):
                        self._result_rows.append(
                            [combined.get(n) for n in self._out_names])
                        found = True
                        right_matched.add(ri)
                if not found and self.join_type in ('LEFT', 'FULL'):
                    self._result_rows.append(
                        [l_row.get(n) for n in self._out_names])
        self.left.close()

        if self.join_type in ('RIGHT', 'FULL'):
            for ri, r_row in enumerate(right_rows):
                if ri not in right_matched:
                    self._result_rows.append(
                        [r_row.get(n) for n in self._out_names])
        self._emitted = False

    def next_batch(self) -> Optional[VectorBatch]:
        if self._emitted: return None
        self._emitted = True
        if not self._result_rows:
            return VectorBatch.empty(self._out_names, self._out_types)
        return VectorBatch.from_rows(self._result_rows, self._out_names, self._out_types)

    def close(self) -> None: pass

    def _eval_cond(self, combined: Dict[str, Any],
                   schema: List[Tuple[str, DataType]]) -> bool:
        cols = {}
        for cn, ct in schema:
            val = combined.get(cn)
            cols[cn] = DataVector.from_scalar(val, ct if val is not None else DataType.INT)
        batch = VectorBatch(columns=cols, _column_order=[n for n, _ in schema], _row_count=1)
        try:
            mask = self._evaluator.evaluate_predicate(self._on_expr, batch)
            return mask.get_bit(0)
        except Exception:
            return False
