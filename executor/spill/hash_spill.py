from __future__ import annotations
"""Hash partition spill — partitions data to temp files by hash."""
from typing import Any, Dict, List
from executor.spill.manager import SpillManager
from executor.spill.temp_file import TempFile
from metal.hash import murmur3_64


class HashSpill:
    """Partitions rows by hash key into temp files."""

    def __init__(self, num_partitions: int = 32,
                 manager: SpillManager = None) -> None:
        self._num_parts = num_partitions
        self._manager = manager or SpillManager()
        self._files: List[TempFile] = []
        for _ in range(num_partitions):
            path = self._manager.create_file(prefix='z1db_hs_')
            self._files.append(TempFile(path))

    def spill_row(self, key: Any, row: list) -> None:
        h = murmur3_64(str(key).encode()) % self._num_parts
        self._files[h].write_row(row)

    def spill_rows(self, key_fn: Any, rows: List[list]) -> None:
        for row in rows:
            key = key_fn(row)
            self.spill_row(key, row)

    def read_partition(self, partition: int) -> List[list]:
        return self._files[partition].read_all()

    def cleanup(self) -> None:
        for f in self._files: f.delete()
        self._manager.cleanup()

    @property
    def num_partitions(self) -> int: return self._num_parts
