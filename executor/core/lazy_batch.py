from __future__ import annotations

"""Lazy batch — defers materialization until absolutely necessary.
Paper: Abadi et al., 2007 "Materialization Strategies"

Filter produces (original_batch, position_bitmap) instead of copying data.
Project only materializes the columns it actually needs."""
from typing import Any, Dict, List, Optional
from executor.core.batch import VectorBatch
from executor.core.vector import DataVector
from metal.bitmap import Bitmap
from storage.types import DataType


class LazyBatch:
    """A batch that defers row filtering until materialization.

    Holds a reference to the original batch + a selection bitmap.
    No data is copied until materialize() or a downstream operator needs it.
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
    def column_names(self) -> List[str]:
        return self._batch.column_names

    @property
    def columns(self) -> Dict[str, DataVector]:
        # Force materialization on direct column access
        return self.materialize().columns

    @property
    def original(self) -> VectorBatch:
        return self._batch

    @property
    def mask(self) -> Bitmap:
        return self._mask

    def get_indices(self) -> list:
        """Lazily compute the list of surviving row indices."""
        if self._indices is None:
            self._indices = self._mask.to_indices()
        return self._indices

    def get_column(self, name: str) -> DataVector:
        """Get a single column, filtered by mask. More efficient than full materialize."""
        if self._materialized is not None:
            return self._materialized.columns[name]
        vec = self._batch.columns[name]
        return vec.filter_by_indices(self.get_indices())

    def materialize(self) -> VectorBatch:
        """Force full materialization. Only do this once."""
        if self._materialized is None:
            self._materialized = self._batch.filter_by_bitmap(self._mask)
        return self._materialized

    def to_rows(self) -> List[list]:
        return self.materialize().to_rows()

    def add_column(self, name: str, vec: DataVector) -> None:
        # Adding a column forces materialization
        m = self.materialize()
        m.add_column(name, vec)
        self._materialized = m


def is_lazy(batch: Any) -> bool:
    """Check if a batch is lazy (deferred materialization)."""
    return isinstance(batch, LazyBatch)


def ensure_batch(batch: Any) -> VectorBatch:
    """Ensure we have a concrete VectorBatch, materializing if needed."""
    if isinstance(batch, LazyBatch):
        return batch.materialize()
    return batch
