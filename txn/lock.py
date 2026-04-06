from __future__ import annotations
"""表级读写锁。写锁等待读者释放后获取。"""
import threading
from typing import Dict


class TableLockManager:
    """表级锁管理器。读锁共享，写锁排他。"""

    def __init__(self) -> None:
        self._mutex = threading.Lock()
        self._readers: Dict[str, int] = {}
        self._writers: Dict[str, bool] = {}
        self._cond = threading.Condition(self._mutex)

    def acquire_read(self, table: str) -> None:
        """获取读锁。等待写锁释放。"""
        with self._cond:
            while self._writers.get(table, False):
                self._cond.wait()
            self._readers[table] = self._readers.get(table, 0) + 1

    def release_read(self, table: str) -> None:
        with self._cond:
            count = self._readers.get(table, 0)
            if count > 1:
                self._readers[table] = count - 1
            else:
                self._readers.pop(table, None)
            self._cond.notify_all()

    def acquire_write(self, table: str) -> None:
        """获取写锁。等待所有读者和写者释放。"""
        with self._cond:
            while (self._writers.get(table, False)
                   or self._readers.get(table, 0) > 0):
                self._cond.wait()
            self._writers[table] = True

    def release_write(self, table: str) -> None:
        with self._cond:
            self._writers[table] = False
            self._cond.notify_all()

    def is_locked_for_write(self, table: str) -> bool:
        with self._mutex:
            return self._writers.get(table, False)

    def reader_count(self, table: str) -> int:
        with self._mutex:
            return self._readers.get(table, 0)
