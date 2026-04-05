from __future__ import annotations
"""Write-Ahead Log — crash recovery via sequential log replay.
Simplified ARIES: every DML writes to WAL before modifying data.
Paper: Mohan et al., 1992 "ARIES"."""
import json
import os
import time
from typing import Any, Dict, List, Optional


class WALEntry:
    __slots__ = ('lsn', 'timestamp', 'op_type', 'table', 'data')

    def __init__(self, lsn: int, op_type: str, table: str,
                 data: Optional[dict] = None) -> None:
        self.lsn = lsn
        self.timestamp = time.time()
        self.op_type = op_type  # INSERT, UPDATE, DELETE, CREATE, DROP, CHECKPOINT
        self.table = table
        self.data = data or {}

    def to_json(self) -> str:
        return json.dumps({
            'lsn': self.lsn,
            'ts': self.timestamp,
            'op': self.op_type,
            'table': self.table,
            'data': self.data,
        }, default=str)

    @staticmethod
    def from_json(line: str) -> WALEntry:
        d = json.loads(line)
        entry = WALEntry(
            lsn=d['lsn'],
            op_type=d['op'],
            table=d['table'],
            data=d.get('data', {}))
        entry.timestamp = d.get('ts', 0)
        return entry


class WriteAheadLog:
    """Append-only WAL file for crash recovery."""

    def __init__(self, data_dir: str) -> None:
        self._data_dir = data_dir
        self._path = os.path.join(data_dir, 'z1db.wal')
        self._lsn = 0
        self._checkpoint_lsn = 0
        self._fh: Optional[Any] = None
        self._load_state()

    def _load_state(self) -> None:
        """Find the highest LSN in existing WAL."""
        if os.path.exists(self._path):
            try:
                with open(self._path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        entry = WALEntry.from_json(line)
                        self._lsn = max(self._lsn, entry.lsn)
                        if entry.op_type == 'CHECKPOINT':
                            self._checkpoint_lsn = entry.lsn
            except Exception:
                pass

    def open(self) -> None:
        os.makedirs(self._data_dir, exist_ok=True)
        self._fh = open(self._path, 'a')

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None

    def append(self, op_type: str, table: str,
               data: Optional[dict] = None) -> int:
        """Write a WAL entry. Returns LSN."""
        self._lsn += 1
        entry = WALEntry(self._lsn, op_type, table, data)
        if self._fh is None:
            self.open()
        self._fh.write(entry.to_json() + '\n')
        self._fh.flush()
        os.fsync(self._fh.fileno())
        return self._lsn

    def checkpoint(self) -> int:
        """Write checkpoint marker. Data up to this point is safe on disk."""
        lsn = self.append('CHECKPOINT', '', {'checkpoint_lsn': self._lsn})
        self._checkpoint_lsn = lsn
        return lsn

    def recover(self) -> List[WALEntry]:
        """Read entries after last checkpoint for recovery."""
        entries = []
        if not os.path.exists(self._path):
            return entries
        with open(self._path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = WALEntry.from_json(line)
                    if entry.lsn > self._checkpoint_lsn:
                        if entry.op_type != 'CHECKPOINT':
                            entries.append(entry)
                except Exception:
                    continue
        return entries

    def truncate_before(self, lsn: int) -> None:
        """Remove entries before LSN (after successful checkpoint)."""
        if not os.path.exists(self._path):
            return
        kept = []
        with open(self._path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = WALEntry.from_json(line)
                    if entry.lsn >= lsn:
                        kept.append(line)
                except Exception:
                    continue
        with open(self._path, 'w') as f:
            for line in kept:
                f.write(line + '\n')

    @property
    def current_lsn(self) -> int:
        return self._lsn

    @property
    def checkpoint_lsn(self) -> int:
        return self._checkpoint_lsn
