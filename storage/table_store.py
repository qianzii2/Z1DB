from __future__ import annotations
"""In-memory columnar store for a single table."""
from typing import TYPE_CHECKING, List, Set
from metal.config import CHUNK_SIZE
from storage.column_chunk import ColumnChunk
from utils.errors import ColumnNotFoundError

if TYPE_CHECKING:
    from catalog.catalog import TableSchema

class TableStore:
    def __init__(self, schema: TableSchema) -> None:
        self.schema = schema
        self._chunks: dict[str, list[ColumnChunk]] = {}
        self.row_count = 0
        for col in schema.columns:
            self._chunks[col.name] = [ColumnChunk(col.dtype)]

    def append_row(self, row: list) -> None:
        first_col = self.schema.columns[0].name
        if self._chunks[first_col][-1].row_count >= CHUNK_SIZE:
            for col in self.schema.columns:
                self._chunks[col.name].append(ColumnChunk(col.dtype))
        for i, col in enumerate(self.schema.columns):
            self._chunks[col.name][-1].append(row[i])
        self.row_count += 1

    def append_rows(self, rows: list[list]) -> int:
        for row in rows: self.append_row(row)
        return len(rows)

    def get_chunk_count(self) -> int:
        return len(self._chunks[self.schema.columns[0].name])

    def get_column_chunks(self, column_name: str) -> list[ColumnChunk]:
        if column_name not in self._chunks: raise ColumnNotFoundError(column_name)
        return self._chunks[column_name]

    def truncate(self) -> None:
        for col in self.schema.columns:
            self._chunks[col.name] = [ColumnChunk(col.dtype)]
        self.row_count = 0

    def read_all_rows(self) -> List[list]:
        """Read all rows as a list of lists (schema-ordered)."""
        rows: List[list] = []
        for chunk_idx in range(self.get_chunk_count()):
            first = self._chunks[self.schema.columns[0].name][chunk_idx]
            for row_idx in range(first.row_count):
                row = []
                for col in self.schema.columns:
                    row.append(self._chunks[col.name][chunk_idx].get(row_idx))
                rows.append(row)
        return rows

    def delete_rows(self, indices_to_delete: Set[int]) -> int:
        """Delete rows by global row index. Returns count deleted."""
        if not indices_to_delete: return 0
        all_rows = self.read_all_rows()
        remaining = [r for i, r in enumerate(all_rows) if i not in indices_to_delete]
        deleted = len(all_rows) - len(remaining)
        self.truncate()
        for r in remaining: self.append_row(r)
        return deleted

    def update_rows(self, indices: Set[int], col_idx: int, new_values: dict) -> int:
        """Update column values for specified rows. new_values: {row_index: value}."""
        all_rows = self.read_all_rows()
        for idx, val in new_values.items():
            if 0 <= idx < len(all_rows):
                all_rows[idx][col_idx] = val
        self.truncate()
        for r in all_rows: self.append_row(r)
        return len(new_values)
