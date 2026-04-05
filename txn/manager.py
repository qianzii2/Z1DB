from __future__ import annotations
"""事务管理器 — BEGIN/COMMIT/ROLLBACK + 自动提交。
修复：auto_begin 后 auto_commit 属性正确返回 True。"""
import threading
from typing import Any, Dict, List, Optional
from txn.lock import TableLockManager


class Transaction:
    __slots__ = ('txn_id', 'modified_tables', 'snapshots', 'active',
                 'auto_started')

    def __init__(self, txn_id: int, auto_started: bool = False) -> None:
        self.txn_id = txn_id
        self.modified_tables: Dict[str, Any] = {}
        self.snapshots: Dict[str, list] = {}
        self.active = True
        self.auto_started = auto_started


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
        """当前事务是否为自动开启的（DML自动提交模式）。
        显式 BEGIN 的事务返回 False，auto_begin 开启的返回 True。"""
        if self._current_txn is None:
            return False
        return self._current_txn.auto_started

    def begin(self) -> int:
        """显式 BEGIN — 需要用户手动 COMMIT/ROLLBACK。"""
        with self._mutex:
            txn_id = self._next_txn_id
            self._next_txn_id += 1
            txn = Transaction(txn_id, auto_started=False)
            self._active[txn_id] = txn
            self._current_txn = txn
            return txn_id

    def commit(self) -> bool:
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
        if self._current_txn is None:
            return False
        txn = self._current_txn
        # 恢复快照（在 mutex 外操作 store 避免死锁）
        if catalog and txn.snapshots:
            for table, snapshot_rows in txn.snapshots.items():
                try:
                    store = catalog.get_store(table)
                    store.truncate()
                    for row in snapshot_rows:
                        store.append_row(list(row))
                except Exception:
                    pass
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

    def snapshot_table(self, table: str, catalog: Any) -> None:
        """DML 前对表做快照（仅首次）。"""
        if self._current_txn is None:
            return
        txn = self._current_txn
        if table not in txn.snapshots:
            store = catalog.get_store(table)
            txn.snapshots[table] = [list(row) for row in store.read_all_rows()]
            txn.modified_tables[table] = True
            self._lock_manager.acquire_write(table)

    def auto_begin(self) -> Optional[int]:
        """自动开启事务（DML 隐式调用）。已在事务中则不重复开启。"""
        if self._current_txn is None:
            with self._mutex:
                txn_id = self._next_txn_id
                self._next_txn_id += 1
                txn = Transaction(txn_id, auto_started=True)
                self._active[txn_id] = txn
                self._current_txn = txn
                return txn_id
        return None

    def auto_commit_if_needed(self, auto_txn_id: Optional[int]) -> None:
        if (auto_txn_id is not None and self._current_txn
                and self._current_txn.txn_id == auto_txn_id):
            self.commit()
