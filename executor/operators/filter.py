from __future__ import annotations
"""Filter (WHERE) operator."""

from typing import Any, List, Optional

from executor.core.batch import VectorBatch
from executor.core.operator import Operator
from executor.expression.evaluator import ExpressionEvaluator
from storage.types import DataType


class FilterOperator(Operator):
    """Filters rows by evaluating a predicate expression."""

    def __init__(self, child: Operator, predicate: Any) -> None:
        super().__init__()
        self.child = child
        self.children = [child]
        self._predicate = predicate
        self._evaluator = ExpressionEvaluator()
        self._closed = False

    def output_schema(self) -> List[tuple[str, DataType]]:
        return self.child.output_schema()

    def open(self) -> None:
        self._closed = False
        self.child.open()

    def close(self) -> None:
        if not self._closed:
            self.child.close()
            self._closed = True

    def next_batch(self) -> Optional[VectorBatch]:
        while True:
            batch = self.child.next_batch()
            if batch is None:
                return None
            mask = self._evaluator.evaluate_predicate(self._predicate, batch)
            filtered = batch.filter_by_bitmap(mask)
            if filtered.row_count > 0:
                return filtered
