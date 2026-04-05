from __future__ import annotations
"""SSTable — Sorted String Table for LSM-Tree persistence.
Immutable sorted file with sparse index and bloom filter."""
import json
import os
import struct
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

try:
    from structures.bloom_filter import BloomFilter
    _HAS_BLOOM = True
except ImportError:
    _HAS_BLOOM = False

SSTABLE_MAGIC = b'Z1SS'
SPARSE_INDEX_INTERVAL = 128


class SSTableWriter:
    """Write sorted entries to an SSTable file."""

    def __init__(self, path: str, schema_columns: List[str]) -> None:
        self._path = path
        self._columns = schema_columns
        self._entries: List[Tuple[Any, Optional[list]]] = []
        self._bloom: Optional[BloomFilter] = None

    def add(self, key: Any, row: Optional[list]) -> None:
        self._entries.append((key, row))

    def finish(self) -> dict:
        """Write all entries to file. Returns metadata dict."""
        n = len(self._entries)
        if _HAS_BLOOM:
            self._bloom = BloomFilter(max(n, 1), 0.01)
            for k, _ in self._entries:
                if k is not None:
                    self._bloom.add(k)

        sparse_index: List[Tuple[Any, int]] = []

        with open(self._path, 'w') as f:
            for i, (key, row) in enumerate(self._entries):
                if i % SPARSE_INDEX_INTERVAL == 0:
                    sparse_index.append((key, f.tell()))
                entry = json.dumps({'k': key, 'v': row}, default=str)
                f.write(entry + '\n')

        meta = {
            'path': self._path,
            'count': n,
            'min_key': self._entries[0][0] if self._entries else None,
            'max_key': self._entries[-1][0] if self._entries else None,
            'columns': self._columns,
            'sparse_index': sparse_index,
        }
        # Write metadata sidecar
        meta_path = self._path + '.meta'
        with open(meta_path, 'w') as f:
            json.dump(meta, f)
        return meta


class SSTableReader:
    """Read from an SSTable file."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._meta: Optional[dict] = None
        self._bloom: Optional[BloomFilter] = None
        self._load_meta()

    def _load_meta(self) -> None:
        meta_path = self._path + '.meta'
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                self._meta = json.load(f)

    @property
    def count(self) -> int:
        return self._meta.get('count', 0) if self._meta else 0

    @property
    def min_key(self) -> Any:
        return self._meta.get('min_key') if self._meta else None

    @property
    def max_key(self) -> Any:
        return self._meta.get('max_key') if self._meta else None

    def might_contain(self, key: Any) -> bool:
        """Bloom filter check. False → definitely not here."""
        if self._bloom:
            return self._bloom.contains(key)
        # No bloom → might contain
        if self._meta:
            mk, xk = self._meta.get('min_key'), self._meta.get('max_key')
            if mk is not None and xk is not None:
                try:
                    return mk <= key <= xk
                except TypeError:
                    pass
        return True

    def get(self, key: Any) -> Optional[list]:
        """Point lookup. O(n) scan (sparse index future optimization)."""
        for k, v in self.scan():
            if k == key:
                return v
            if k is not None and key is not None:
                try:
                    if k > key:
                        break
                except TypeError:
                    pass
        return None

    def scan(self) -> Iterator[Tuple[Any, Optional[list]]]:
        """Full scan in sorted order."""
        if not os.path.exists(self._path):
            return
        with open(self._path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                yield (entry['k'], entry['v'])

    def scan_range(self, lo: Any, hi: Any) -> Iterator[Tuple[Any, Optional[list]]]:
        """Range scan [lo, hi]."""
        for k, v in self.scan():
            if k is None:
                continue
            try:
                if k < lo:
                    continue
                if k > hi:
                    break
            except TypeError:
                continue
            yield (k, v)

    def delete_files(self) -> None:
        """Remove SSTable and metadata files."""
        for path in (self._path, self._path + '.meta'):
            try:
                os.unlink(path)
            except OSError:
                pass
