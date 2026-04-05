from __future__ import annotations
"""Temp file wrapper for spill operations."""
import json, os
from typing import Any, Iterator, List


class TempFile:
    """Append-only temp file for spilling rows."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._count = 0

    def write_row(self, row: list) -> None:
        with open(self._path, 'a') as f:
            f.write(json.dumps(row, default=str) + '\n')
        self._count += 1

    def write_rows(self, rows: List[list]) -> None:
        with open(self._path, 'a') as f:
            for row in rows:
                f.write(json.dumps(row, default=str) + '\n')
        self._count += len(rows)

    def read_all(self) -> List[list]:
        rows = []
        try:
            with open(self._path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line: rows.append(json.loads(line))
        except FileNotFoundError: pass
        return rows

    def iterator(self) -> Iterator[list]:
        try:
            with open(self._path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line: yield json.loads(line)
        except FileNotFoundError: pass

    @property
    def count(self) -> int: return self._count

    def delete(self) -> None:
        try: os.unlink(self._path)
        except OSError: pass
