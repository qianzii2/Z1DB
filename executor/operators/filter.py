from __future__ import annotations
"""Filter (WHERE) operator — supports lazy materialization."""
from typing import Any, List, Optional
from executor.core.batch import VectorBatch
from executor.core.operator import Operator
from executor.expression.evaluator import ExpressionEvaluator
from storage.types import DataType

try:
    from executor.core.lazy_batch import LazyBatch
    _HAS_LAZY = True
except ImportError:
    _HAS_LAZY = False


class FilterOperator(Operator):
    """Filters rows by evaluating a predicate expression.
    Outputs LazyBatch when possible to defer materialization."""

    def __init__(self, child: Operator, predicate: Any,
                 lazy: bool = True) -> None:
        super().__init__()
        self.child = child
        self.children = [child]
        self._predicate = predicate
        self._evaluator = ExpressionEvaluator()
        self._closed = False
        self._lazy = lazy and _HAS_LAZY

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
            # Handle incoming LazyBatch
            if _HAS_LAZY and isinstance(batch, LazyBatch):
                batch = batch.materialize()
            mask = self._evaluator.evaluate_predicate(self._predicate, batch)
            count = mask.popcount()
            if count == 0:
                continue
            # If all rows pass, return original batch
            if count == batch.row_count:
                return batch
            # Lazy: return (batch, mask) pair without copying
            if self._lazy:
                return LazyBatch(batch, mask)
            # Eager: materialize immediately
            filtered = batch.filter_by_bitmap(mask)
            if filtered.row_count > 0:
                return filtered
