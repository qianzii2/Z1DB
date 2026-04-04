from __future__ import annotations
"""ExecutionResult — the final output of query execution."""

from dataclasses import dataclass, field
from typing import List, Optional

from executor.core.batch import VectorBatch
from storage.types import DataType


@dataclass
class ExecutionResult:
    columns: List[str] = field(default_factory=list)
    column_types: List[DataType] = field(default_factory=list)
    rows: List[list] = field(default_factory=list)
    row_count: int = 0
    affected_rows: int = 0
    message: str = ''
    timing: float = 0.0

    @staticmethod
    def from_batch(batch: VectorBatch,
                   col_types: Optional[List[DataType]] = None) -> ExecutionResult:
        columns = batch.column_names
        rows = batch.to_rows()
        if col_types is None:
            col_types = [batch.columns[n].dtype for n in columns] if columns else []
        return ExecutionResult(
            columns=columns,
            column_types=col_types,
            rows=rows,
            row_count=len(rows),
        )
