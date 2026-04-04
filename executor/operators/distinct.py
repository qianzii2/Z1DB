from __future__ import annotations
"""DISTINCT operator — removes duplicate rows."""
from typing import List, Optional, Tuple
from executor.core.batch import VectorBatch
from executor.core.operator import Operator
from storage.types import DataType


class DistinctOperator(Operator):
    """Pipeline breaker that deduplicates rows using a set."""

    def __init__(self, child: Operator) -> None:
        super().__init__()
        self.child = child
        self.children = [child]
        self._result: Optional[VectorBatch] = None
        self._emitted = False

    def output_schema(self) -> List[Tuple[str, DataType]]:
        return self.child.output_schema()

    def open(self) -> None:
        self.child.open()
        seen: set = set()
        unique_rows: list = []
        col_names = None
        col_types = None

        while True:
            batch = self.child.next_batch()
            if batch is None:
                break
            if col_names is None:
                col_names = batch.column_names
                col_types = [batch.columns[n].dtype for n in col_names]
            for i in range(batch.row_count):
                row = tuple(batch.columns[n].get(i) for n in col_names)
                # NULL equality for DISTINCT: NULLs are considered equal
                key = tuple(_null_safe(v) for v in row)
                if key not in seen:
                    seen.add(key)
                    unique_rows.append(list(row))

        self.child.close()

        if col_names and unique_rows:
            self._result = VectorBatch.from_rows(unique_rows, col_names, col_types)
        else:
            schema = self.output_schema()
            self._result = VectorBatch.empty(
                [n for n, _ in schema], [t for _, t in schema])
        self._emitted = False

    def next_batch(self) -> Optional[VectorBatch]:
        if self._emitted or self._result is None:
            return None
        self._emitted = True
        return self._result

    def close(self) -> None:
        pass


def _null_safe(v: object) -> object:
    """Wrap None so that it hashes and compares equal to other Nones."""
    if v is None:
        return _NULL_SENTINEL
    return v

class _NullSentinel:
    def __hash__(self) -> int: return 0
    def __eq__(self, other: object) -> bool: return isinstance(other, _NullSentinel)

_NULL_SENTINEL = _NullSentinel()
