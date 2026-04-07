from __future__ import annotations
"""嵌套循环连接 — O(n*m) 暴力扫描。NANO 层级回退方案。
[D06] 条件评估委托到 join_utils。"""
from typing import Any, Dict, List, Optional, Tuple
from executor.core.batch import VectorBatch
from executor.core.operator import Operator
from executor.operators.join.join_utils import eval_join_condition
from storage.types import DataType


class NestedLoopJoinOperator(Operator):
    """简单嵌套循环连接，适合极小表（<64行）。"""

    def __init__(self, left: Operator, right: Operator,
                 join_type: str, on_expr: Any) -> None:
        super().__init__()
        self.left = left; self.right = right
        self.children = [left, right]
        self._join_type = join_type
        self._on_expr = on_expr
        self._result_rows: list = []
        self._out_names: list = []
        self._out_types: list = []
        self._emitted = False

    def output_schema(self) -> List[Tuple[str, DataType]]:
        if self._join_type in ('SEMI', 'ANTI'):
            return self.left.output_schema()
        return self.left.output_schema() + self.right.output_schema()

    def open(self) -> None:
        self.left.open(); self.right.open()
        schema = self.output_schema()
        self._out_names = [n for n, _ in schema]
        self._out_types = [t for _, t in schema]
        left_schema = self.left.output_schema()
        left_col_count = len(left_schema)
        # 完整 schema（用于条件评估）
        full_schema = self.left.output_schema() + self.right.output_schema()

        # 物化右表
        right_rows: list = []
        while True:
            b = self._next_child_batch(self.right)
            if b is None: break
            for i in range(b.row_count):
                right_rows.append(
                    {n: b.columns[n].get(i) for n in b.column_names})
        self.right.close()

        self._result_rows = []
        right_matched: set = set()

        while True:
            lb = self._next_child_batch(self.left)
            if lb is None: break
            for li in range(lb.row_count):
                l_row = {n: lb.columns[n].get(li) for n in lb.column_names}
                found = False
                for ri, r_row in enumerate(right_rows):
                    combined = {**l_row, **r_row}
                    # [D06] 委托到 join_utils
                    if (self._on_expr is None
                            or eval_join_condition(combined, full_schema, self._on_expr)):
                        if self._join_type == 'SEMI':
                            self._result_rows.append(
                                [l_row.get(n) for n in self._out_names])
                            found = True; break
                        elif self._join_type == 'ANTI':
                            found = True; break
                        else:
                            self._result_rows.append(
                                [combined.get(n) for n in self._out_names])
                            found = True; right_matched.add(ri)
                if not found:
                    if self._join_type in ('LEFT', 'FULL'):
                        row = [l_row.get(n) for n in self._out_names[:left_col_count]]
                        row += [None] * (len(self._out_names) - left_col_count)
                        self._result_rows.append(row)
                    elif self._join_type == 'ANTI':
                        self._result_rows.append(
                            [l_row.get(n) for n in self._out_names])
        self.left.close()

        if self._join_type in ('RIGHT', 'FULL'):
            for ri, r_row in enumerate(right_rows):
                if ri not in right_matched:
                    row = [None] * left_col_count
                    row += [r_row.get(n) for n in self._out_names[left_col_count:]]
                    self._result_rows.append(row)
        self._emitted = False

    def next_batch(self) -> Optional[VectorBatch]:
        if self._emitted: return None
        self._emitted = True
        if not self._result_rows:
            return VectorBatch.empty(self._out_names, self._out_types)
        return VectorBatch.from_rows(
            self._result_rows, self._out_names, self._out_types)

    def close(self) -> None:
        try:
            self.left.close()
        except Exception:
            pass
        try:
            self.right.close()
        except Exception:
            pass

