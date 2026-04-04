from __future__ import annotations
"""In-memory columnar store for a single table."""

from typing import TYPE_CHECKING

from metal.config import CHUNK_SIZE
from storage.column_chunk import ColumnChunk
from utils.errors import ColumnNotFoundError

if TYPE_CHECKING:
    from catalog.catalog import TableSchema


class TableStore:
    """Manages all column chunks for one table."""

    def __init__(self, schema: TableSchema) -> None:
        self.schema = schema
        self._chunks: dict[str, list[ColumnChunk]] = {}
        self.row_count = 0
        for col in schema.columns:
            self._chunks[col.name] = [ColumnChunk(col.dtype)]

    # ------------------------------------------------------------------
    def append_row(self, row: list) -> None:
        """Append one row (values ordered by schema)."""
        # If current chunks are full, create new ones
        first_col = self.schema.columns[0].name
        if self._chunks[first_col][-1].row_count >= CHUNK_SIZE:
            for col in self.schema.columns:
                self._chunks[col.name].append(ColumnChunk(col.dtype))

        for i, col in enumerate(self.schema.columns):
            self._chunks[col.name][-1].append(row[i])
        self.row_count += 1

    def append_rows(self, rows: list[list]) -> int:
        for row in rows:
            self.append_row(row)
        return len(rows)

    # ------------------------------------------------------------------
    def get_chunk_count(self) -> int:
        first_col = self.schema.columns[0].name
        return len(self._chunks[first_col])

    def get_column_chunks(self, column_name: str) -> list[ColumnChunk]:
        if column_name not in self._chunks:
            raise ColumnNotFoundError(column_name)
        return self._chunks[column_name]

    def truncate(self) -> None:
        for col in self.schema.columns:
            self._chunks[col.name] = [ColumnChunk(col.dtype)]
        self.row_count = 0
