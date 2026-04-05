from __future__ import annotations
"""Delta Store — row-oriented append buffer for writes.
Paper: Grund et al., 2010 "HYRISE"

INSERT → append to delta.
UPDATE → mark old row deleted + append new row to delta.
DELETE → mark row deleted.
Periodically merged into Main Store by background worker."""
from typing import Any, Dict, List, Optional, Set
from metal.bitmap import Bitmap


class DeltaStore:
    """Append-only write buffer. Stores rows in insertion order."""

    __slots__ = ('_rows', '_column_names', '_delete_bitmap', '_next_id')

    def __init__(self, column_names: List[str]) -> None:
        self._column_names = column_names
        self._rows: List[list] = []
        self._delete_bitmap = Bitmap(0)
        self._next_id = 0

    @property
    def row_count(self) -> int:
        return len(self._rows)

    @property
    def active_row_count(self) -> int:
        """Rows not marked as deleted."""
        return self.row_count - self._delete_bitmap.popcount()

    def insert(self, row: list) -> int:
        """Append a row. Returns row_id within delta."""
        row_id = self._next_id
        self._rows.append(list(row))
        self._delete_bitmap.ensure_capacity(row_id + 1)
        self._next_id += 1
        return row_id

    def mark_deleted(self, row_id: int) -> None:
        """Mark a row as deleted (tombstone)."""
        if 0 <= row_id < len(self._rows):
            self._delete_bitmap.set_bit(row_id)

    def is_deleted(self, row_id: int) -> bool:
        return self._delete_bitmap.get_bit(row_id)

    def get_row(self, row_id: int) -> Optional[list]:
        if row_id < 0 or row_id >= len(self._rows):
            return None
        if self._delete_bitmap.get_bit(row_id):
            return None
        return self._rows[row_id]

    def scan_active(self) -> List[list]:
        """Scan all non-deleted rows."""
        result = []
        for i, row in enumerate(self._rows):
            if not self._delete_bitmap.get_bit(i):
                result.append(row)
        return result

    def drain(self) -> List[list]:
        """Extract all active rows and clear the delta. Used during merge."""
        active = self.scan_active()
        self.clear()
        return active

    def clear(self) -> None:
        self._rows.clear()
        self._delete_bitmap = Bitmap(0)
        self._next_id = 0
