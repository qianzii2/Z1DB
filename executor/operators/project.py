from __future__ import annotations
"""Projection operator."""

from typing import Any, Dict, List, Optional, Tuple

from executor.core.batch import VectorBatch
from executor.core.operator import Operator
from executor.expression.evaluator import ExpressionEvaluator
from storage.types import DataType


class ProjectOperator(Operator):
    """Evaluates a list of expressions to produce output columns."""

    def __init__(self, child: Operator,
                 projections: List[Tuple[str, Any]]) -> None:
        super().__init__()
        self.child = child
        self.children = [child]
        self._projections = projections  # [(output_name, ast_expr), ...]
        self._evaluator = ExpressionEvaluator()
        self._closed = False

    def output_schema(self) -> List[tuple[str, DataType]]:
        child_schema: Dict[str, DataType] = dict(self.child.output_schema())
        result = []
        for name, expr in self._projections:
            dt = ExpressionEvaluator.infer_type(expr, child_schema)
            result.append((name, dt))
        return result

    def open(self) -> None:
        self._closed = False
        self.child.open()

    def close(self) -> None:
        if not self._closed:
            self.child.close()
            self._closed = True

    def next_batch(self) -> Optional[VectorBatch]:
        batch = self.child.next_batch()
        if batch is None:
            return None
        result_cols = {}
        names: List[str] = []
        for name, expr in self._projections:
            result_cols[name] = self._evaluator.evaluate(expr, batch)
            names.append(name)
        return VectorBatch(columns=result_cols, _column_order=names)
