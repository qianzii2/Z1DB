from __future__ import annotations
"""Sequential scan operator — reads all chunks from a TableStore."""

from typing import List, Optional

from executor.core.batch import VectorBatch
from executor.core.operator import Operator
from executor.core.vector import DataVector
from storage.table_store import TableStore
from storage.types import DataType


class SeqScan(Operator):
    """Full table scan, chunk by chunk."""

    def __init__(self, table_name: str, store: TableStore,
                 columns: List[str]) -> None:
        super().__init__()
        self._table_name = table_name
        self._store = store
        self._columns = columns
        self._current_chunk = 0

    def output_schema(self) -> List[tuple[str, DataType]]:
        col_map = {c.name: c.dtype for c in self._store.schema.columns}
        return [(n, col_map[n]) for n in self._columns]

    def open(self) -> None:
        self._current_chunk = 0

    def close(self) -> None:
        pass

    def next_batch(self) -> Optional[VectorBatch]:
        while self._current_chunk < self._store.get_chunk_count():
            chunk_idx = self._current_chunk
            self._current_chunk += 1
            # Skip empty chunks (e.g. freshly-created table)
            first_col = self._columns[0] if self._columns else None
            if first_col:
                chunks = self._store.get_column_chunks(first_col)
                if chunks[chunk_idx].row_count == 0:
                    continue

            cols = {}
            for name in self._columns:
                chunk_list = self._store.get_column_chunks(name)
                cols[name] = DataVector.from_column_chunk(chunk_list[chunk_idx])
            return VectorBatch(columns=cols, _column_order=list(self._columns))
        return None
