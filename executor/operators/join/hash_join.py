from __future__ import annotations
"""Hash join operator — nested-loop with hash-like evaluation."""
from typing import Any, Dict, List, Optional, Tuple
from executor.core.batch import VectorBatch
from executor.core.operator import Operator
from executor.core.vector import DataVector
from executor.expression.evaluator import ExpressionEvaluator
from metal.bitmap import Bitmap
from storage.types import DataType


class HashJoinOperator(Operator):
    """Hash join: materialises right, probes from left."""

    def __init__(self, left: Operator, right: Operator, join_type: str,
                 on_expr: Any) -> None:
        super().__init__()
        self.left = left
        self.right = right
        self.children = [left, right]
        self.join_type = join_type  # INNER, LEFT, RIGHT
        self._on_expr = on_expr
        self._evaluator = ExpressionEvaluator()
        self._result_rows: List[list] = []
        self._emitted = False

    def output_schema(self) -> List[Tuple[str, DataType]]:
        ls = self.left.output_schema()
        rs = self.right.output_schema()
        if self.join_type == 'RIGHT':
            return rs + ls
        return ls + rs

    def open(self) -> None:
        self.left.open()
        self.right.open()

        # Materialize right side as list of dicts
        right_schema = self.right.output_schema()
        right_col_names = [n for n, _ in right_schema]
        right_rows_data: List[Dict[str, Any]] = []
        while True:
            b = self.right.next_batch()
            if b is None:
                break
            for i in range(b.row_count):
                row = {n: b.columns[n].get(i) for n in b.column_names}
                right_rows_data.append(row)
        self.right.close()

        left_schema = self.left.output_schema()
        left_col_names = [n for n, _ in left_schema]
        out_schema = self.output_schema()
        out_names = [n for n, _ in out_schema]
        out_types = [t for _, t in out_schema]

        self._result_rows = []
        right_matched: set = set()

        while True:
            lb = self.left.next_batch()
            if lb is None:
                break
            for li in range(lb.row_count):
                l_row = {n: lb.columns[n].get(li) for n in lb.column_names}
                found_match = False
                for ri, r_row in enumerate(right_rows_data):
                    combined = {**l_row, **r_row}
                    if self._on_expr is None or self._eval_cond(combined, out_schema):
                        out_row = [combined.get(n) for n in out_names]
                        self._result_rows.append(out_row)
                        found_match = True
                        right_matched.add(ri)
                if not found_match and self.join_type == 'LEFT':
                    out_row = []
                    for n in out_names:
                        out_row.append(l_row.get(n))
                    self._result_rows.append(out_row)
        self.left.close()

        if self.join_type == 'RIGHT':
            for ri, r_row in enumerate(right_rows_data):
                if ri not in right_matched:
                    out_row = [r_row.get(n) for n in out_names]
                    self._result_rows.append(out_row)

        self._out_names = out_names
        self._out_types = out_types
        self._emitted = False

    def next_batch(self) -> Optional[VectorBatch]:
        if self._emitted:
            return None
        self._emitted = True
        if not self._result_rows:
            return VectorBatch.empty(self._out_names, self._out_types)
        return VectorBatch.from_rows(self._result_rows, self._out_names, self._out_types)

    def close(self) -> None:
        pass

    def _eval_cond(self, combined: Dict[str, Any],
                   schema: List[Tuple[str, DataType]]) -> bool:
        cols: Dict[str, DataVector] = {}
        for cn, ct in schema:
            val = combined.get(cn)
            dt = ct if val is not None else DataType.INT
            cols[cn] = DataVector.from_scalar(val, dt)
        batch = VectorBatch(columns=cols,
                            _column_order=[n for n, _ in schema], _row_count=1)
        try:
            mask = self._evaluator.evaluate_predicate(self._on_expr, batch)
            return mask.get_bit(0)
        except Exception:
            return False
