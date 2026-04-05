from __future__ import annotations
"""External Sort — handles data larger than memory.
Generate sorted runs → K-way merge with LoserTree."""
import tempfile
import json
import os
from typing import Any, Iterator, List, Optional, Tuple
from structures.tournament_tree import LoserTree


class ExternalSort:
    """Disk-based sort for datasets exceeding memory limit."""

    def __init__(self, memory_limit: int = 64 * 1024 * 1024,
                 temp_dir: Optional[str] = None) -> None:
        self._memory_limit = memory_limit
        self._temp_dir = temp_dir or tempfile.gettempdir()
        self._run_files: list = []

    def sort(self, rows: list, key_fn: Any = None) -> list:
        """Sort rows, spilling to disk if needed.
        Estimated memory per row: 200 bytes (conservative)."""
        estimated_mem = len(rows) * 200
        if estimated_mem <= self._memory_limit:
            rows.sort(key=key_fn)
            return rows

        # Phase 1: Generate sorted runs
        chunk_size = max(1, self._memory_limit // 200)
        self._run_files = []
        for start in range(0, len(rows), chunk_size):
            end = min(start + chunk_size, len(rows))
            chunk = rows[start:end]
            chunk.sort(key=key_fn)
            self._write_run(chunk)

        # Phase 2: K-way merge
        return self._merge_runs(key_fn)

    def _write_run(self, sorted_chunk: list) -> None:
        fd, path = tempfile.mkstemp(suffix='.run', dir=self._temp_dir)
        try:
            with os.fdopen(fd, 'w') as f:
                for row in sorted_chunk:
                    f.write(json.dumps(row) + '\n')
        except Exception:
            os.close(fd)
            raise
        self._run_files.append(path)

    def _merge_runs(self, key_fn: Any) -> list:
        if not self._run_files:
            return []

        # Open all run files as iterators
        file_handles = []
        iterators = []
        for path in self._run_files:
            fh = open(path, 'r')
            file_handles.append(fh)
            iterators.append(self._run_iterator(fh))

        try:
            # Use LoserTree for K-way merge
            if key_fn:
                tree = LoserTree(iterators, key_fn=key_fn)
            else:
                tree = LoserTree(iterators)
            result = tree.merge_all()
        finally:
            for fh in file_handles:
                fh.close()
            self._cleanup()

        return result

    @staticmethod
    def _run_iterator(fh: Any) -> Iterator:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)

    def _cleanup(self) -> None:
        for path in self._run_files:
            try:
                os.unlink(path)
            except OSError:
                pass
        self._run_files = []
