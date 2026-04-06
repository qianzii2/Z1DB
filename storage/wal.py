from __future__ import annotations
"""Write-Ahead Log — 崩溃恢复 + 自动轮转。
[M02] 支持 DDL 操作（CREATE/DROP/ALTER TABLE）。
使用原子 rename 防止截断时数据丢失。"""
import json
import os
import time
from typing import Any, Dict, List, Optional


class WALEntry:
    """WAL 条目：LSN + 时间戳 + 操作类型 + 表名 + 数据。"""

    __slots__ = ('lsn', 'timestamp', 'op_type', 'table', 'data')

    def __init__(self, lsn: int, op_type: str, table: str,
                 data: Optional[dict] = None) -> None:
        self.lsn = lsn
        self.timestamp = time.time()
        self.op_type = op_type
        self.table = table
        self.data = data or {}

    def to_json(self) -> str:
        return json.dumps({
            'lsn': self.lsn, 'ts': self.timestamp,
            'op': self.op_type, 'table': self.table,
            'data': self.data,
        }, default=str)

    @staticmethod
    def from_json(line: str) -> 'WALEntry':
        d = json.loads(line)
        entry = WALEntry(
            lsn=d['lsn'], op_type=d['op'],
            table=d['table'], data=d.get('data', {}))
        entry.timestamp = d.get('ts', 0)
        return entry


# [M02] 支持的操作类型
# DML: INSERT, UPDATE, DELETE
# DDL: CREATE_TABLE, DROP_TABLE, ALTER_TABLE
# 系统: CHECKPOINT


class WriteAheadLog:
    """追加写入 WAL。checkpoint 时 fsync + 原子轮转。"""

    MAX_WAL_BYTES = 10 * 1024 * 1024  # 10 MB 后轮转

    def __init__(self, data_dir: str) -> None:
        self._data_dir = data_dir
        self._path = os.path.join(data_dir, 'z1db.wal')
        self._lsn = 0
        self._checkpoint_lsn = 0
        self._fh: Optional[Any] = None
        self._dirty = False
        self._load_state()

    def _load_state(self) -> None:
        """从现有 WAL 文件恢复 LSN 和 checkpoint 位置。"""
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
            if self._dirty:
                self._fsync()
            self._fh.close()
            self._fh = None

    def append(self, op_type: str, table: str,
               data: Optional[dict] = None) -> int:
        """追加一条 WAL 记录。返回 LSN。"""
        self._lsn += 1
        entry = WALEntry(self._lsn, op_type, table, data)
        if self._fh is None:
            self.open()
        self._fh.write(entry.to_json() + '\n')
        self._fh.flush()
        self._dirty = True
        return self._lsn

    def checkpoint(self) -> int:
        """写入 CHECKPOINT 标记 + fsync + 可能轮转。"""
        lsn = self.append('CHECKPOINT', '',
                           {'checkpoint_lsn': self._lsn})
        self._fsync()
        self._checkpoint_lsn = lsn
        self._maybe_rotate()
        return lsn

    def _fsync(self) -> None:
        if self._fh:
            try:
                os.fsync(self._fh.fileno())
                self._dirty = False
            except OSError:
                pass

    def _maybe_rotate(self) -> None:
        """WAL 文件过大时截断旧条目。"""
        try:
            if not os.path.exists(self._path):
                return
            if os.path.getsize(self._path) > self.MAX_WAL_BYTES:
                self.truncate_before(self._checkpoint_lsn)
        except OSError:
            pass

    def recover(self) -> List[WALEntry]:
        """恢复：读取 checkpoint 之后的所有条目。"""
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
        """原子截断：先写临时文件 → fsync → rename。"""
        if not os.path.exists(self._path):
            return
        was_open = self._fh is not None
        if was_open:
            self._fh.close()
            self._fh = None
        # 收集要保留的条目
        kept = []
        try:
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
        except Exception:
            if was_open:
                self._fh = open(self._path, 'a')
            return
        # 原子写入
        tmp_path = self._path + '.tmp'
        try:
            with open(tmp_path, 'w') as f:
                for line in kept:
                    f.write(line + '\n')
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self._path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        if was_open:
            self._fh = open(self._path, 'a')

    @property
    def current_lsn(self) -> int:
        return self._lsn

    @property
    def checkpoint_lsn(self) -> int:
        return self._checkpoint_lsn
