from __future__ import annotations

"""Lazy batch — defers materialization until needed.
Filter produces (batch, mask) instead of copying all column data.
Reduces 50% of intermediate data copying."""
from typing import Any, Dict, List, Optional
from executor.core.batch import VectorBatch
from executor.core.vector import DataVector
from metal.bitmap import Bitmap
from storage.types import DataType


class LazyBatch:
    """Wraps a VectorBatch + selection bitmap without materializing.

    Only materializes (copies data) when explicitly requested or
    when a downstream operator cannot handle lazy batches.
    """

    __slots__ = ('_batch', '_mask', '_indices', '_materialized')

    def __init__(self, batch: VectorBatch, mask: Bitmap) -> None:
        self._batch = batch
        self._mask = mask
        self._indices: Optional[list] = None
        self._materialized: Optional[VectorBatch] = None

    @property
    def row_count(self) -> int:
        return self._mask.popcount()

    @property
    def column_names(self) -> list:
        return self._batch.column_names

    @property
    def columns(self) -> dict:
        return self.materialize().columns

    def get_indices(self) -> list:
        """Lazily compute selected row indices."""
        if self._indices is None:
            self._indices = self._mask.to_indices()
        return self._indices

    def materialize(self) -> VectorBatch:
        """Materialize into a concrete VectorBatch. Only copies data once."""
        if self._materialized is None:
            self._materialized = self._batch.filter_by_bitmap(self._mask)
        return self._materialized

    def get_column_lazy(self, name: str, expr_evaluator: Any = None,
                        expr: Any = None) -> DataVector:
        """Get a column, applying filter lazily.
        If expr is provided, evaluate on original batch then filter."""
        if expr is not None and expr_evaluator is not None:
            # Evaluate on full batch, then filter — avoids double materialization
            full_vec = expr_evaluator.evaluate(expr, self._batch)
            return full_vec.filter_by_indices(self.get_indices())
        # Simple column access
        vec = self._batch.get_column(name)
        return vec.filter_by_indices(self.get_indices())
