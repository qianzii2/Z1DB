from __future__ import annotations
"""哈希分区溢写。"""
from typing import Any, Dict, List
from executor.spill.manager import SpillManager
from executor.spill.temp_file import TempFile
from metal.hash import z1hash64


class HashSpill:
    def __init__(self, num_partitions: int = 32,
                 manager: SpillManager = None) -> None:
        self._num_parts = num_partitions
        self._manager = manager or SpillManager()
        self._files: List[TempFile] = []
        for _ in range(num_partitions):
            path = self._manager.create_file(
                prefix='z1db_hs_')
            self._files.append(TempFile(path))

    def spill_row(self, key: Any, row: list) -> None:
        h = z1hash64(
            str(key).encode()) % self._num_parts
        self._files[h].write_row(row)

    def spill_rows(self, key_fn, rows):
        for row in rows:
            self.spill_row(key_fn(row), row)

    def read_partition(self,
                       partition: int) -> List[list]:
        return self._files[partition].read_all()

    def cleanup(self):
        for f in self._files: f.delete()
        self._manager.cleanup()

    @property
    def num_partitions(self): return self._num_parts
