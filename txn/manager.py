from __future__ import annotations
"""事务管理器 — BEGIN/COMMIT/ROLLBACK + 自动提交。
修复：auto_begin 创建的事务通过 _auto_txn_id 追踪，commit/rollback 正确释放。"""
import threading
from typing import Any, Dict, List, Optional
from txn.lock import TableLockManager


class Transaction:
    __slots__ = ('txn_id', 'modified_tables', 'snapshots', 'active',
                 'is_auto')

    def __init__(self, txn_id: int, is_auto: bool = False) -> None:
        self.txn_id = txn_id
        self.modified_tables: Dict[str, Any] = {}
        self.snapshots: Dict[str, list] = {}
        self.active = True
        self.is_auto = is_auto  # 是否由 auto_begin 创建


class TransactionManager:
    def __init__(self) -> None:
        self._lock_manager = TableLockManager()
        self._next_txn_id = 1
        self._active: Dict[int, Transaction] = {}
        self._mutex = threading.Lock()
        self._current_txn: Optional[Transaction] = None

    @property
    def lock_manager(self) -> TableLockManager:
        return self._lock_manager

    @property
    def in_transaction(self) -> bool:
        return self._current_txn is not None and self._current_txn.active

    @property
    def auto_commit(self) -> bool:
        """当前事务是否为自动事务（可自动提交）。"""
        if self._current_txn is None:
            return False
        return self._current_txn.is_auto

    def begin(self) -> int:
        """显式 BEGIN — 非自动事务。"""
        with self._mutex:
            txn_id = self._next_txn_id
            self._next_txn_id += 1
            txn = Transaction(txn_id, is_auto=False)
            self._active[txn_id] = txn
            self._current_txn = txn
            return txn_id

    def commit(self) -> bool:
        """提交当前事务，释放所有写锁。"""
        if self._current_txn is None:
            return False
        txn = self._current_txn
        with self._mutex:
            for table in txn.modified_tables:
                try:
                    self._lock_manager.release_write(table)
                except Exception:
                    pass
            txn.active = False
            self._active.pop(txn.txn_id, None)
            self._current_txn = None
        return True

    def rollback(self, catalog: Any = None) -> bool:
        """回滚当前事务：恢复快照 + 释放写锁。"""
        if self._current_txn is None:
            return False
        txn = self._current_txn
        # 先在 mutex 内标记非活跃，防止并发问题
        with self._mutex:
            txn.active = False
            self._active.pop(txn.txn_id, None)
            self._current_txn = None
        # 恢复快照（mutex 外操作存储，避免死锁）
        if catalog and txn.snapshots:
            for table, snapshot_rows in txn.snapshots.items():
                try:
                    store = catalog.get_store(table)
                    store.truncate()
                    for row in snapshot_rows:
                        store.append_row(list(row))
                except Exception:
                    pass
        # 释放写锁
        for table in txn.modified_tables:
            try:
                self._lock_manager.release_write(table)
            except Exception:
                pass
        return True

    def snapshot_table(self, table: str, catalog: Any) -> None:
        """对表做快照（仅首次）。"""
        if self._current_txn is None:
            return
        txn = self._current_txn
        if table not in txn.snapshots:
            store = catalog.get_store(table)
            txn.snapshots[table] = [list(row) for row in store.read_all_rows()]
            txn.modified_tables[table] = True
            self._lock_manager.acquire_write(table)

    def auto_begin(self) -> Optional[int]:
        """自动事务：如果没有活跃事务则创建，返回事务 ID。"""
        if self._current_txn is not None:
            return None  # 已有显式事务，不创建
        with self._mutex:
            txn_id = self._next_txn_id
            self._next_txn_id += 1
            txn = Transaction(txn_id, is_auto=True)
            self._active[txn_id] = txn
            self._current_txn = txn
            return txn_id

    def auto_commit_if_needed(self, auto_txn_id: Optional[int]) -> None:
        """如果 auto_txn_id 对应当前自动事务，则提交。"""
        if (auto_txn_id is not None
                and self._current_txn is not None
                and self._current_txn.txn_id == auto_txn_id
                and self._current_txn.is_auto):
            self.commit()
