from __future__ import annotations
"""表级读写锁。
修复 M15：acquire_write 等待读者释放。"""
import threading
from typing import Dict


class TableLockManager:
    """表级锁，写锁等待读者释放后获取。"""

    def __init__(self) -> None:
        self._mutex = threading.Lock()
        self._readers: Dict[str, int] = {}
        self._writers: Dict[str, bool] = {}
        self._cond = threading.Condition(self._mutex)

    def acquire_read(self, table: str) -> None:
        with self._cond:
            # 读者不阻塞其他读者，但要等写者释放
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
        with self._cond:
            # M15 修复：等待所有写者和读者释放
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
