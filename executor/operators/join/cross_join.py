from __future__ import annotations
"""Cross join operator."""
from typing import Any, Dict, List, Optional, Tuple
from executor.core.batch import VectorBatch
from executor.core.operator import Operator
from storage.types import DataType


class CrossJoinOperator(Operator):
    """Cartesian product of left × right."""

    def __init__(self, left: Operator, right: Operator) -> None:
        super().__init__()
        self.left = left
        self.right = right
        self.children = [left, right]
        self._result_rows: List[list] = []
        self._emitted = False

    def output_schema(self) -> List[Tuple[str, DataType]]:
        return self.left.output_schema() + self.right.output_schema()

    def open(self) -> None:
        self.left.open()
        self.right.open()
        right_rows: list = []
        r_names = [n for n, _ in self.right.output_schema()]
        while True:
            b = self.right.next_batch()
            if b is None:
                break
            for i in range(b.row_count):
                right_rows.append([b.columns[n].get(i) for n in b.column_names])
        self.right.close()

        self._result_rows = []
        while True:
            lb = self.left.next_batch()
            if lb is None:
                break
            for li in range(lb.row_count):
                l_vals = [lb.columns[n].get(li) for n in lb.column_names]
                for r_vals in right_rows:
                    self._result_rows.append(l_vals + r_vals)
        self.left.close()
        self._emitted = False

    def next_batch(self) -> Optional[VectorBatch]:
        if self._emitted:
            return None
        self._emitted = True
        s = self.output_schema()
        names = [n for n, _ in s]
        types = [t for _, t in s]
        if not self._result_rows:
            return VectorBatch.empty(names, types)
        return VectorBatch.from_rows(self._result_rows, names, types)

    def close(self) -> None:
        pass
