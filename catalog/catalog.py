from __future__ import annotations
"""系统目录 — Schema 注册 + 存储管理 + 索引管理。
持久化模式使用 LSM-Tree，内存模式使用 TableStore。
[B01] 修复重复的 alter_drop_column。
[B02] 实现缺失的 alter_add_column。"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
from storage.table_store import TableStore
from storage.types import DataType
from utils.errors import (DuplicateError, TableNotFoundError,
                           ColumnNotFoundError, ExecutionError)

try:
    from catalog.index_manager import IndexManager
    _HAS_INDEX = True
except ImportError:
    _HAS_INDEX = False

try:
    from storage.lsm.lsm_store import LSMStore
    _HAS_LSM = True
except ImportError:
    _HAS_LSM = False


@dataclass
class ColumnSchema:
    """列定义。"""
    name: str
    dtype: DataType
    nullable: bool = True
    primary_key: bool = False
    max_length: Optional[int] = None


@dataclass
class TableSchema:
    """表定义。"""
    name: str
    columns: List[ColumnSchema] = field(default_factory=list)

    @property
    def primary_key(self) -> Optional[str]:
        for c in self.columns:
            if c.primary_key:
                return c.name
        return None

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]


class Catalog:
    """系统目录。管理所有表的 schema 和存储引擎。"""

    def __init__(self, data_dir: str = ':memory:') -> None:
        self._schemas: dict[str, TableSchema] = {}
        self._stores: dict[str, object] = {}
        self.data_dir = data_dir
        self._index_manager: Optional[IndexManager] = (
            IndexManager() if _HAS_INDEX else None)
        if data_dir != ':memory:':
            self._load()

    @property
    def is_persistent(self) -> bool:
        return self.data_dir != ':memory:'

    @property
    def use_lsm(self) -> bool:
        """持久化模式且 LSM 可用时使用 LSM。"""
        return self.is_persistent and _HAS_LSM

    @property
    def index_manager(self) -> Optional[IndexManager]:
        return self._index_manager

    def get_table_columns(self, table: str) -> list[str]:
        if table not in self._schemas:
            raise TableNotFoundError(table)
        return self._schemas[table].column_names

    def table_exists(self, table: str) -> bool:
        return table in self._schemas

    # ═══ DDL ═══

    def create_table(self, schema: TableSchema,
                     if_not_exists: bool = False) -> None:
        if schema.name in self._schemas:
            if if_not_exists:
                return
            raise DuplicateError(
                f"table '{schema.name}' already exists")
        self._schemas[schema.name] = schema
        if self.use_lsm:
            self._stores[schema.name] = LSMStore(
                schema, self.data_dir)
        else:
            self._stores[schema.name] = TableStore(schema)
        if self.is_persistent:
            self._save_schema()

    def drop_table(self, name: str,
                   if_exists: bool = False) -> None:
        if name not in self._schemas:
            if if_exists:
                return
            raise TableNotFoundError(name)
        store = self._stores.get(name)
        if isinstance(store, LSMStore):
            store.close()
        del self._schemas[name]
        del self._stores[name]
        if self._index_manager:
            self._index_manager.invalidate_table(name)
        if self.is_persistent:
            self._save_schema()
            import shutil
            lsm_dir = Path(self.data_dir) / f'lsm_{name}'
            if lsm_dir.exists():
                shutil.rmtree(lsm_dir, ignore_errors=True)
            from catalog import persist
            persist.remove_table_data(
                name, Path(self.data_dir))

    def get_table(self, name: str) -> TableSchema:
        if name not in self._schemas:
            raise TableNotFoundError(name)
        return self._schemas[name]

    def get_store(self, name: str) -> object:
        if name not in self._stores:
            raise TableNotFoundError(name)
        return self._stores[name]

    def list_tables(self) -> list[str]:
        return list(self._schemas.keys())

    # ═══ ALTER TABLE [B01][B02] ═══

    def alter_add_column(self, table: str,
                         col_schema: ColumnSchema) -> None:
        """[B02] 添加列。新列所有现有行填充 NULL。"""
        if table not in self._schemas:
            raise TableNotFoundError(table)
        schema = self._schemas[table]
        if any(c.name == col_schema.name
               for c in schema.columns):
            raise DuplicateError(
                f"column '{col_schema.name}' already exists")
        store = self._stores[table]
        # LSMStore：先 flush
        if self.use_lsm and hasattr(store, 'flush'):
            store.flush()
        all_rows = store.read_all_rows()
        schema.columns.append(col_schema)
        # 重建存储，每行末尾追加 NULL
        self._rebuild_store(table, schema, all_rows,
                            add_column=True)
        if self.is_persistent:
            self._save_schema()

    def alter_drop_column(self, table: str,
                          col_name: str) -> None:
        """[B01] 删除列。仅保留一个实现。"""
        if table not in self._schemas:
            raise TableNotFoundError(table)
        schema = self._schemas[table]
        idx = None
        for i, c in enumerate(schema.columns):
            if c.name == col_name:
                idx = i
                break
        if idx is None:
            raise ColumnNotFoundError(col_name)
        if len(schema.columns) <= 1:
            raise ExecutionError(
                "cannot drop the only column")
        store = self._stores[table]
        # LSMStore：先 flush
        if self.use_lsm and hasattr(store, 'flush'):
            store.flush()
        all_rows = store.read_all_rows()
        schema.columns.pop(idx)
        for row in all_rows:
            if idx < len(row):
                row.pop(idx)
        self._rebuild_store(table, schema, all_rows)
        if self.is_persistent:
            self._save_schema()

    def alter_rename_column(self, table: str,
                            old_name: str,
                            new_name: str) -> None:
        if table not in self._schemas:
            raise TableNotFoundError(table)
        schema = self._schemas[table]
        if any(c.name == new_name for c in schema.columns):
            raise DuplicateError(
                f"column '{new_name}' already exists")
        if not any(c.name == old_name
                   for c in schema.columns):
            raise ColumnNotFoundError(old_name)
        store = self._stores[table]
        all_rows = store.read_all_rows()
        for c in schema.columns:
            if c.name == old_name:
                c.name = new_name
                break
        self._rebuild_store(table, schema, all_rows)
        if self.is_persistent:
            self._save_schema()

    def _rebuild_store(self, table: str,
                       schema: TableSchema,
                       rows: List[list],
                       add_column: bool = False) -> None:
        """重建存储（ALTER 后调用）。"""
        old_store = self._stores.get(table)
        if isinstance(old_store, LSMStore):
            old_store.close()
        if self.use_lsm:
            import shutil
            lsm_dir = Path(self.data_dir) / f'lsm_{table}'
            if lsm_dir.exists():
                shutil.rmtree(lsm_dir, ignore_errors=True)
            new_store = LSMStore(schema, self.data_dir)
        else:
            new_store = TableStore(schema)
        self._stores[table] = new_store
        for row in rows:
            if add_column:
                row.append(None)
            new_store.append_row(row)
        if isinstance(new_store, LSMStore):
            new_store.flush()

    # ═══ 持久化 ═══

    def persist(self) -> None:
        """持久化。LSM 模式只存 schema，数据已在 SSTable 中。"""
        if not self.is_persistent:
            return
        self._save_schema()
        if self.use_lsm:
            for name, store in self._stores.items():
                if isinstance(store, LSMStore):
                    store.flush()
        else:
            from catalog import persist as _persist
            _persist.save_data(
                self._stores, Path(self.data_dir))

    def _persist(self) -> None:
        self.persist()

    def _save_schema(self) -> None:
        from catalog import persist as _persist
        _persist.save(
            self._schemas,
            Path(self.data_dir) / 'catalog.json')

    def _load(self) -> None:
        path = Path(self.data_dir) / 'catalog.json'
        if not path.exists():
            return
        from catalog import persist as _persist
        self._schemas = _persist.load(path)
        for name, schema in self._schemas.items():
            lsm_dir = Path(self.data_dir) / f'lsm_{name}'
            if self.use_lsm and lsm_dir.exists():
                self._stores[name] = LSMStore(
                    schema, self.data_dir)
            elif self.use_lsm:
                old_store = TableStore(schema)
                _persist.load_data(
                    {name: old_store}, Path(self.data_dir))
                if old_store.row_count > 0:
                    lsm = LSMStore(schema, self.data_dir)
                    for row in old_store.read_all_rows():
                        lsm.append_row(row)
                    lsm.flush()
                    self._stores[name] = lsm
                else:
                    self._stores[name] = LSMStore(
                        schema, self.data_dir)
            else:
                self._stores[name] = TableStore(schema)
                _persist.load_data(
                    {name: self._stores[name]},
                    Path(self.data_dir))
