from __future__ import annotations
"""Spill manager — manages temporary file allocation for disk spills."""
import os, tempfile
from typing import List


class SpillManager:
    """Manages temp files for sort/join/agg spills."""

    def __init__(self, temp_dir: str = '') -> None:
        self._temp_dir = temp_dir or tempfile.gettempdir()
        self._files: List[str] = []
        self._total_bytes = 0

    def create_file(self, prefix: str = 'z1db_') -> str:
        fd, path = tempfile.mkstemp(prefix=prefix, dir=self._temp_dir)
        os.close(fd)
        self._files.append(path)
        return path

    def cleanup(self) -> None:
        for path in self._files:
            try: os.unlink(path)
            except OSError: pass
        self._files.clear()
        self._total_bytes = 0

    @property
    def num_files(self) -> int: return len(self._files)

    @property
    def total_bytes(self) -> int: return self._total_bytes
