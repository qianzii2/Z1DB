from __future__ import annotations
"""嵌套循环连接 — O(n*m)暴力扫描。NANO层级回退方案。"""
from typing import Any, Dict, List, Optional, Tuple
from executor.core.batch import VectorBatch
from executor.core.operator import Operator
from executor.core.vector import DataVector
from executor.expression.evaluator import ExpressionEvaluator
from storage.types import DataType


class NestedLoopJoinOperator(Operator):
    """简单嵌套循环连接，适合极小表。"""

    def __init__(self, left: Operator, right: Operator,
                 join_type: str, on_expr: Any) -> None:
        super().__init__()
        self.left = left
        self.right = right
        self.children = [left, right]
        self._join_type = join_type
        self._on_expr = on_expr
        self._evaluator = ExpressionEvaluator()
        self._result_rows: list = []
        self._out_names: list = []
        self._out_types: list = []
        self._emitted = False

    def output_schema(self) -> List[Tuple[str, DataType]]:
        return self.left.output_schema() + self.right.output_schema()

    def open(self) -> None:
        self.left.open()
        self.right.open()
        schema = self.output_schema()
        self._out_names = [n for n, _ in schema]
        self._out_types = [t for _, t in schema]
        left_col_count = len(self.left.output_schema())

        # 物化右表
        right_rows: list = []
        while True:
            b = self._ensure_batch(self.right.next_batch())
            if b is None:
                break
            for i in range(b.row_count):
                right_rows.append(
                    {n: b.columns[n].get(i) for n in b.column_names})
        self.right.close()

        self._result_rows = []
        right_matched: set = set()

        while True:
            lb = self._ensure_batch(self.left.next_batch())
            if lb is None:
                break
            for li in range(lb.row_count):
                l_row = {n: lb.columns[n].get(li) for n in lb.column_names}
                found = False
                for ri, r_row in enumerate(right_rows):
                    combined = {**l_row, **r_row}
                    if self._on_expr is None or self._eval_cond(combined):
                        out = [combined.get(n) for n in self._out_names]
                        self._result_rows.append(out)
                        found = True
                        right_matched.add(ri)
                if not found and self._join_type in ('LEFT', 'FULL'):
                    row = [l_row.get(n)
                           for n in self._out_names[:left_col_count]]
                    row += [None] * (len(self._out_names) - left_col_count)
                    self._result_rows.append(row)
        self.left.close()

        if self._join_type in ('RIGHT', 'FULL'):
            for ri, r_row in enumerate(right_rows):
                if ri not in right_matched:
                    row = [None] * left_col_count
                    row += [r_row.get(n)
                            for n in self._out_names[left_col_count:]]
                    self._result_rows.append(row)
        self._emitted = False

    def next_batch(self) -> Optional[VectorBatch]:
        if self._emitted:
            return None
        self._emitted = True
        if not self._result_rows:
            return VectorBatch.empty(self._out_names, self._out_types)
        return VectorBatch.from_rows(
            self._result_rows, self._out_names, self._out_types)

    def close(self) -> None:
        pass

    def _eval_cond(self, combined: Dict[str, Any]) -> bool:
        schema = self.output_schema()
        cols = {}
        for cn, ct in schema:
            val = combined.get(cn)
            cols[cn] = DataVector.from_scalar(
                val, ct if val is not None else DataType.INT)
        batch = VectorBatch(
            columns=cols,
            _column_order=[n for n, _ in schema], _row_count=1)
        try:
            mask = self._evaluator.evaluate_predicate(self._on_expr, batch)
            return mask.get_bit(0)
        except Exception:
            return False
