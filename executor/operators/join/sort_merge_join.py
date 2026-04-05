from __future__ import annotations
"""排序归并连接 — 双方排序后O(n+m)归并。
接口统一：接受on_expr AST，自动提取等值key。"""
from typing import Any, Dict, List, Optional, Tuple
from executor.core.batch import VectorBatch
from executor.core.operator import Operator
from executor.core.vector import DataVector
from executor.expression.evaluator import ExpressionEvaluator
from executor.operators.join.hash_join import extract_equi_keys
from storage.types import DataType


class SortMergeJoinOperator(Operator):
    """排序归并连接。双方按join key排序后线性归并。"""

    def __init__(self, left: Operator, right: Operator,
                 join_type: str, on_expr: Any) -> None:
        super().__init__()
        self.left = left
        self.right = right
        self.children = [left, right]
        self._join_type = join_type
        self._on_expr = on_expr
        self._result_rows: list = []
        self._out_names: list = []
        self._out_types: list = []
        self._emitted = False
        # 从on_expr提取等值key
        left_key, right_key = extract_equi_keys(on_expr)
        self._left_key = left_key or ''
        self._right_key = right_key or ''

    def output_schema(self) -> List[Tuple[str, DataType]]:
        return self.left.output_schema() + self.right.output_schema()

    def open(self) -> None:
        self.left.open()
        self.right.open()
        schema = self.output_schema()
        self._out_names = [n for n, _ in schema]
        self._out_types = [t for _, t in schema]
        l_names = [n for n, _ in self.left.output_schema()]
        r_names = [n for n, _ in self.right.output_schema()]
        left_col_count = len(l_names)

        # 收集并排序双方
        l_rows = self._collect_and_sort(self.left, l_names, self._left_key)
        r_rows = self._collect_and_sort(self.right, r_names, self._right_key)
        self.left.close()
        self.right.close()

        # 归并
        self._result_rows = []
        li = ri = 0
        right_matched: set = set()

        while li < len(l_rows) and ri < len(r_rows):
            lk = l_rows[li][0]
            rk = r_rows[ri][0]

            if lk is None:
                if self._join_type in ('LEFT', 'FULL'):
                    row = l_rows[li][1] + [None] * len(r_names)
                    self._result_rows.append(row)
                li += 1
                continue
            if rk is None:
                if self._join_type in ('RIGHT', 'FULL'):
                    row = [None] * left_col_count + r_rows[ri][1]
                    self._result_rows.append(row)
                ri += 1
                continue

            try:
                cmp = (lk > rk) - (lk < rk)
            except TypeError:
                cmp = (str(lk) > str(rk)) - (str(lk) < str(rk))

            if cmp < 0:
                if self._join_type in ('LEFT', 'FULL'):
                    row = l_rows[li][1] + [None] * len(r_names)
                    self._result_rows.append(row)
                li += 1
            elif cmp > 0:
                if self._join_type in ('RIGHT', 'FULL'):
                    row = [None] * left_col_count + r_rows[ri][1]
                    self._result_rows.append(row)
                ri += 1
            else:
                # 等值 — 处理重复
                li_start = li
                while li < len(l_rows) and l_rows[li][0] == lk:
                    li += 1
                ri_start = ri
                while ri < len(r_rows) and r_rows[ri][0] == rk:
                    ri += 1
                for lj in range(li_start, li):
                    for rj in range(ri_start, ri):
                        self._result_rows.append(
                            l_rows[lj][1] + r_rows[rj][1])
                        right_matched.add(rj)

        # LEFT/FULL：剩余左行
        if self._join_type in ('LEFT', 'FULL'):
            while li < len(l_rows):
                row = l_rows[li][1] + [None] * len(r_names)
                self._result_rows.append(row)
                li += 1

        # RIGHT/FULL：剩余右行
        if self._join_type in ('RIGHT', 'FULL'):
            while ri < len(r_rows):
                if ri not in right_matched:
                    row = [None] * left_col_count + r_rows[ri][1]
                    self._result_rows.append(row)
                ri += 1

        self._emitted = False

    def _collect_and_sort(self, op: Operator, names: list,
                          key_col: str) -> List[Tuple[Any, list]]:
        """收集所有行，返回 (key_value, row_values) 按key排序。"""
        rows = []
        while True:
            batch = self._ensure_batch(op.next_batch())
            if batch is None:
                break
            for i in range(batch.row_count):
                row = [batch.columns[n].get(i) for n in names]
                key_val = None
                if key_col in batch.columns:
                    key_val = batch.columns[key_col].get(i)
                rows.append((key_val, row))
        rows.sort(key=lambda x: (
            x[0] is None, x[0] if x[0] is not None else 0))
        return rows

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
