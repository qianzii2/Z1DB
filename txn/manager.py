from __future__ import annotations
"""Transaction manager — BEGIN/COMMIT/ROLLBACK with auto-commit."""
import threading
from typing import Any, Dict, List, Optional
from txn.lock import TableLockManager


class Transaction:
    __slots__ = ('txn_id', 'modified_tables', 'snapshots', 'active')

    def __init__(self, txn_id: int) -> None:
        self.txn_id = txn_id
        self.modified_tables: Dict[str, Any] = {}
        self.snapshots: Dict[str, list] = {}
        self.active = True


class TransactionManager:
    def __init__(self) -> None:
        self._lock_manager = TableLockManager()
        self._next_txn_id = 1
        self._active: Dict[int, Transaction] = {}
        self._mutex = threading.Lock()
        self._current_txn: Optional[Transaction] = None
        self._auto_commit = True

    @property
    def lock_manager(self) -> TableLockManager:
        return self._lock_manager

    @property
    def in_transaction(self) -> bool:
        return self._current_txn is not None and self._current_txn.active

    @property
    def auto_commit(self) -> bool:
        return self._auto_commit and not self.in_transaction

    def begin(self) -> int:
        with self._mutex:
            txn_id = self._next_txn_id
            self._next_txn_id += 1
            txn = Transaction(txn_id)
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
        # Restore snapshots OUTSIDE mutex to avoid any lock interaction issues
        if catalog and txn.snapshots:
            for table, snapshot_rows in txn.snapshots.items():
                store = catalog.get_store(table)
                store.truncate()
                for row in snapshot_rows:
                    # Deep copy each row to prevent any reference issues
                    store.append_row(list(row))
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
        if self._current_txn is None:
            return
        txn = self._current_txn
        if table not in txn.snapshots:
            store = catalog.get_store(table)
            # Deep copy all rows
            txn.snapshots[table] = [list(row) for row in store.read_all_rows()]
            txn.modified_tables[table] = True
            self._lock_manager.acquire_write(table)

    def auto_begin(self) -> Optional[int]:
        if self._auto_commit and self._current_txn is None:
            return self.begin()
        return None

    def auto_commit_if_needed(self, auto_txn_id: Optional[int]) -> None:
        if auto_txn_id is not None and self._current_txn and \
           self._current_txn.txn_id == auto_txn_id:
            self.commit()
