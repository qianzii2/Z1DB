from __future__ import annotations
"""System catalog — schema registry and store management."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
from storage.table_store import TableStore
from storage.types import DataType
from utils.errors import DuplicateError, TableNotFoundError, ColumnNotFoundError, ExecutionError


@dataclass
class ColumnSchema:
    name: str
    dtype: DataType
    nullable: bool = True
    primary_key: bool = False
    max_length: Optional[int] = None


@dataclass
class TableSchema:
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
    def __init__(self, data_dir: str = ':memory:') -> None:
        self._schemas: dict[str, TableSchema] = {}
        self._stores: dict[str, TableStore] = {}
        self.data_dir = data_dir
        if data_dir != ':memory:':
            self._load()

    @property
    def is_persistent(self) -> bool:
        return self.data_dir != ':memory:'

    def get_table_columns(self, table: str) -> list[str]:
        if table not in self._schemas:
            raise TableNotFoundError(table)
        return self._schemas[table].column_names

    def table_exists(self, table: str) -> bool:
        return table in self._schemas

    def create_table(self, schema: TableSchema, if_not_exists: bool = False) -> None:
        if schema.name in self._schemas:
            if if_not_exists:
                return
            raise DuplicateError(f"table '{schema.name}' already exists")
        self._schemas[schema.name] = schema
        self._stores[schema.name] = TableStore(schema)

    def drop_table(self, name: str, if_exists: bool = False) -> None:
        if name not in self._schemas:
            if if_exists:
                return
            raise TableNotFoundError(name)
        del self._schemas[name]
        del self._stores[name]
        if self.is_persistent:
            from catalog import persist
            persist.remove_table_data(name, Path(self.data_dir))

    def get_table(self, name: str) -> TableSchema:
        if name not in self._schemas:
            raise TableNotFoundError(name)
        return self._schemas[name]

    def get_store(self, name: str) -> TableStore:
        if name not in self._stores:
            raise TableNotFoundError(name)
        return self._stores[name]

    def list_tables(self) -> list[str]:
        return list(self._schemas.keys())

    def alter_add_column(self, table: str, col: ColumnSchema) -> None:
        if table not in self._schemas:
            raise TableNotFoundError(table)
        schema = self._schemas[table]
        if any(c.name == col.name for c in schema.columns):
            raise DuplicateError(f"column '{col.name}' already exists")
        store = self._stores[table]
        all_rows = store.read_all_rows()
        schema.columns.append(col)
        self._stores[table] = TableStore(schema)
        for row in all_rows:
            row.append(None)
            self._stores[table].append_row(row)

    def alter_drop_column(self, table: str, col_name: str) -> None:
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
            raise ExecutionError("cannot drop the only column")
        store = self._stores[table]
        all_rows = store.read_all_rows()
        schema.columns.pop(idx)
        self._stores[table] = TableStore(schema)
        for row in all_rows:
            row.pop(idx)
            self._stores[table].append_row(row)

    def alter_rename_column(self, table: str, old_name: str, new_name: str) -> None:
        if table not in self._schemas:
            raise TableNotFoundError(table)
        schema = self._schemas[table]
        if any(c.name == new_name for c in schema.columns):
            raise DuplicateError(f"column '{new_name}' already exists")
        if not any(c.name == old_name for c in schema.columns):
            raise ColumnNotFoundError(old_name)
        store = self._stores[table]
        all_rows = store.read_all_rows()
        for c in schema.columns:
            if c.name == old_name:
                c.name = new_name
                break
        self._stores[table] = TableStore(schema)
        for row in all_rows:
            self._stores[table].append_row(row)

    def persist(self) -> None:
        """Flush schemas and data to disk. No-op for :memory:."""
        if not self.is_persistent:
            return
        from catalog import persist as _persist
        _persist.save(self._schemas, Path(self.data_dir) / 'catalog.json')
        _persist.save_data(self._stores, Path(self.data_dir))

    # Keep old name for compatibility
    def _persist(self) -> None:
        self.persist()

    def _load(self) -> None:
        path = Path(self.data_dir) / 'catalog.json'
        if path.exists():
            from catalog import persist as _persist
            self._schemas = _persist.load(path)
            for name, schema in self._schemas.items():
                self._stores[name] = TableStore(schema)
            _persist.load_data(self._stores, Path(self.data_dir))
