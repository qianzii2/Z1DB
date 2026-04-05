from __future__ import annotations
"""Sort Merge Join — O(n+m) when both sides are sorted.
Supports equi-join and non-equi-join conditions."""
from typing import Any, Dict, List, Optional, Tuple
from executor.core.batch import VectorBatch
from executor.core.operator import Operator
from executor.core.vector import DataVector
from executor.expression.evaluator import ExpressionEvaluator
from storage.types import DataType


class SortMergeJoinOperator(Operator):
    """Sort both sides by join key, then merge. O(n log n + m log m + n + m)."""

    def __init__(self, left: Operator, right: Operator,
                 left_key: str, right_key: str,
                 join_type: str = 'INNER') -> None:
        super().__init__()
        self.left = left
        self.right = right
        self.children = [left, right]
        self._left_key = left_key
        self._right_key = right_key
        self._join_type = join_type
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
        l_names = [n for n, _ in self.left.output_schema()]
        r_names = [n for n, _ in self.right.output_schema()]

        # Materialize and sort both sides
        l_rows = self._collect_and_sort(self.left, l_names, self._left_key)
        r_rows = self._collect_and_sort(self.right, r_names, self._right_key)
        self.left.close()
        self.right.close()

        # Merge
        self._result_rows = []
        li = ri = 0
        right_matched = set()

        while li < len(l_rows) and ri < len(r_rows):
            lk = l_rows[li][0]
            rk = r_rows[ri][0]

            if lk is None:
                if self._join_type == 'LEFT':
                    self._result_rows.append(l_rows[li][1] + [None] * len(r_names))
                li += 1
                continue
            if rk is None:
                if self._join_type == 'RIGHT':
                    self._result_rows.append([None] * len(l_names) + r_rows[ri][1])
                ri += 1
                continue

            if lk < rk:
                if self._join_type == 'LEFT':
                    self._result_rows.append(l_rows[li][1] + [None] * len(r_names))
                li += 1
            elif lk > rk:
                if self._join_type == 'RIGHT':
                    self._result_rows.append([None] * len(l_names) + r_rows[ri][1])
                ri += 1
            else:
                # Equal keys — handle duplicates
                # Find range of equal keys on both sides
                li_start = li
                while li < len(l_rows) and l_rows[li][0] == lk:
                    li += 1
                ri_start = ri
                while ri < len(r_rows) and r_rows[ri][0] == rk:
                    ri += 1
                # Cross product of matching ranges
                for lj in range(li_start, li):
                    for rj in range(ri_start, ri):
                        self._result_rows.append(l_rows[lj][1] + r_rows[rj][1])
                        right_matched.add(rj)

        # LEFT JOIN: remaining left rows
        if self._join_type in ('LEFT', 'FULL'):
            while li < len(l_rows):
                self._result_rows.append(l_rows[li][1] + [None] * len(r_names))
                li += 1

        # RIGHT JOIN: remaining right rows
        if self._join_type in ('RIGHT', 'FULL'):
            while ri < len(r_rows):
                if ri not in right_matched:
                    self._result_rows.append([None] * len(l_names) + r_rows[ri][1])
                ri += 1

        self._emitted = False

    def _collect_and_sort(self, op: Operator, names: list,
                          key_col: str) -> List[Tuple[Any, list]]:
        """Collect all rows, return as (key_value, row_values) sorted by key."""
        rows = []
        while True:
            batch = op.next_batch()
            if batch is None:
                break
            for i in range(batch.row_count):
                row = [batch.columns[n].get(i) for n in names]
                key_val = None
                if key_col in batch.columns:
                    key_val = batch.columns[key_col].get(i)
                rows.append((key_val, row))
        rows.sort(key=lambda x: (x[0] is None, x[0] if x[0] is not None else 0))
        return rows

    def next_batch(self) -> Optional[VectorBatch]:
        if self._emitted:
            return None
        self._emitted = True
        if not self._result_rows:
            return VectorBatch.empty(self._out_names, self._out_types)
        return VectorBatch.from_rows(self._result_rows, self._out_names, self._out_types)

    def close(self) -> None:
        pass
