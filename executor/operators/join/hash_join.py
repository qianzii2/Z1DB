from __future__ import annotations
"""Hash join — INNER, LEFT, RIGHT, FULL OUTER."""
from typing import Any, Dict, List, Optional, Tuple
from executor.core.batch import VectorBatch
from executor.core.operator import Operator
from executor.core.vector import DataVector
from executor.expression.evaluator import ExpressionEvaluator
from storage.types import DataType


class HashJoinOperator(Operator):
    def __init__(self, left: Operator, right: Operator, join_type: str, on_expr: Any) -> None:
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
        ls = self.left.output_schema(); rs = self.right.output_schema()
        return ls + rs

    def open(self) -> None:
        self.left.open(); self.right.open()
        rs = self.right.output_schema(); ls = self.left.output_schema()
        out_schema = ls + rs
        self._out_names = [n for n, _ in out_schema]
        self._out_types = [t for _, t in out_schema]
        l_names = [n for n, _ in ls]; r_names = [n for n, _ in rs]
        # Materialize right
        right_rows: List[Dict[str, Any]] = []
        while True:
            b = self.right.next_batch()
            if b is None: break
            for i in range(b.row_count):
                right_rows.append({n: b.columns[n].get(i) for n in b.column_names})
        self.right.close()
        self._result_rows = []
        right_matched: set = set()
        left_rows_data: List[Dict[str, Any]] = []
        # Process left
        while True:
            lb = self.left.next_batch()
            if lb is None: break
            for li in range(lb.row_count):
                l_row = {n: lb.columns[n].get(li) for n in lb.column_names}
                left_rows_data.append(l_row)
                found = False
                for ri, r_row in enumerate(right_rows):
                    combined = {**l_row, **r_row}
                    if self._on_expr is None or self._eval_cond(combined, out_schema):
                        self._result_rows.append([combined.get(n) for n in self._out_names])
                        found = True; right_matched.add(ri)
                if not found and self.join_type in ('LEFT', 'FULL'):
                    row = [l_row.get(n) for n in self._out_names]
                    self._result_rows.append(row)
        self.left.close()
        # Right/Full unmatched
        if self.join_type in ('RIGHT', 'FULL'):
            for ri, r_row in enumerate(right_rows):
                if ri not in right_matched:
                    row = [r_row.get(n) for n in self._out_names]
                    self._result_rows.append(row)
        self._emitted = False

    def next_batch(self) -> Optional[VectorBatch]:
        if self._emitted: return None
        self._emitted = True
        if not self._result_rows:
            return VectorBatch.empty(self._out_names, self._out_types)
        return VectorBatch.from_rows(self._result_rows, self._out_names, self._out_types)

    def close(self) -> None: pass

    def _eval_cond(self, combined: Dict[str, Any], schema: List[Tuple[str, DataType]]) -> bool:
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
