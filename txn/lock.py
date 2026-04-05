from __future__ import annotations
"""Table-level read-write locks.
Read-Read: parallel. Read-Write: readers don't block (COW).
Write-Write: serialized."""
import threading
from typing import Dict, Set


class TableLockManager:
    """Table-level locking with read-write semantics."""

    def __init__(self) -> None:
        self._mutex = threading.Lock()
        self._readers: Dict[str, int] = {}
        self._writers: Dict[str, bool] = {}
        self._write_cond = threading.Condition(self._mutex)

    def acquire_read(self, table: str) -> None:
        with self._mutex:
            self._readers[table] = self._readers.get(table, 0) + 1

    def release_read(self, table: str) -> None:
        with self._mutex:
            count = self._readers.get(table, 0)
            if count > 1:
                self._readers[table] = count - 1
            else:
                self._readers.pop(table, None)
            self._write_cond.notify_all()

    def acquire_write(self, table: str) -> None:
        with self._write_cond:
            while self._writers.get(table, False):
                self._write_cond.wait()
            self._writers[table] = True

    def release_write(self, table: str) -> None:
        with self._write_cond:
            self._writers[table] = False
            self._write_cond.notify_all()

    def is_locked_for_write(self, table: str) -> bool:
        with self._mutex:
            return self._writers.get(table, False)

    def reader_count(self, table: str) -> int:
        with self._mutex:
            return self._readers.get(table, 0)
