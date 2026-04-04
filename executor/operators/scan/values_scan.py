from __future__ import annotations
"""DualScan — produces exactly one zero-column row (for SELECT without FROM)."""

from typing import List, Optional

from executor.core.batch import VectorBatch
from executor.core.operator import Operator
from storage.types import DataType


class DualScan(Operator):
    """Produces a single row with no columns (used for ``SELECT 1+1``)."""

    def __init__(self) -> None:
        super().__init__()
        self._emitted = False

    def output_schema(self) -> List[tuple[str, DataType]]:
        return []

    def open(self) -> None:
        self._emitted = False

    def close(self) -> None:
        pass

    def next_batch(self) -> Optional[VectorBatch]:
        if self._emitted:
            return None
        self._emitted = True
        return VectorBatch.single_row_no_columns()
