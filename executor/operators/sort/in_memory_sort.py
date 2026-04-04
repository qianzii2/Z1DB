from __future__ import annotations
"""In-memory sort operator (pipeline breaker)."""

import functools
from typing import Any, List, Optional, Tuple

from executor.core.batch import VectorBatch
from executor.core.operator import Operator
from executor.expression.evaluator import ExpressionEvaluator
from storage.types import DataType


class SortOperator(Operator):
    """Materialises all input, sorts, then emits as a single batch.

    This is a *pipeline breaker*: ``open()`` consumes all child data and
    closes the child.
    """

    def __init__(self, child: Operator,
                 sort_keys: List[Tuple[Any, str, Optional[str]]]) -> None:
        super().__init__()
        self.child = child
        self.children = [child]
        # sort_keys: [(expr, 'ASC'|'DESC', 'NULLS_FIRST'|'NULLS_LAST'|None)]
        self._sort_keys = sort_keys
        self._result: Optional[VectorBatch] = None
        self._emitted = False
        self._child_closed = False

    def output_schema(self) -> List[tuple[str, DataType]]:
        return self.child.output_schema()

    def open(self) -> None:
        self.child.open()
        batches: List[VectorBatch] = []
        while True:
            b = self.child.next_batch()
            if b is None:
                break
            batches.append(b)
        self.child.close()
        self._child_closed = True

        if not batches:
            self._result = None
            self._emitted = True
            return

        merged = VectorBatch.merge(batches)
        evaluator = ExpressionEvaluator()

        key_columns: List[Tuple[list, str, str]] = []
        for expr, direction, nulls_pos in self._sort_keys:
            if nulls_pos is None:
                nulls_pos = 'NULLS_LAST' if direction == 'ASC' else 'NULLS_FIRST'
            vec = evaluator.evaluate(expr, merged)
            key_columns.append((vec.to_python_list(), direction, nulls_pos))

        indices = list(range(merged.row_count))
        indices.sort(key=functools.cmp_to_key(
            lambda i, j: self._compare_rows(i, j, key_columns)))
        self._result = merged.reorder_by_indices(indices)
        self._emitted = False

    def next_batch(self) -> Optional[VectorBatch]:
        if self._emitted or self._result is None:
            return None
        self._emitted = True
        return self._result

    def close(self) -> None:
        if not self._child_closed:
            self.child.close()
            self._child_closed = True

    # ------------------------------------------------------------------
    @staticmethod
    def _compare_rows(i: int, j: int,
                      key_columns: List[Tuple[list, str, str]]) -> int:
        for values, direction, nulls_pos in key_columns:
            cmp = SortOperator._compare_values(values[i], values[j], nulls_pos)
            if cmp != 0:
                return -cmp if direction == 'DESC' else cmp
        return 0

    @staticmethod
    def _compare_values(a: Any, b: Any, nulls_pos: str) -> int:
        a_null = a is None
        b_null = b is None
        if a_null and b_null:
            return 0
        if a_null:
            return 1 if nulls_pos == 'NULLS_LAST' else -1
        if b_null:
            return -1 if nulls_pos == 'NULLS_LAST' else 1
        try:
            if a < b:
                return -1
            if a > b:
                return 1
            return 0
        except TypeError:
            # Fallback: compare by type name (should not happen in practice)
            ta, tb = type(a).__name__, type(b).__name__
            if ta < tb:
                return -1
            if ta > tb:
                return 1
            return 0
