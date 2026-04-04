from __future__ import annotations
"""System catalog — schema registry and store management."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from storage.table_store import TableStore
from storage.types import DataType
from utils.errors import DuplicateError, TableNotFoundError


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
    """Central registry of table schemas and their backing stores."""

    def __init__(self, data_dir: str = ':memory:') -> None:
        self._schemas: dict[str, TableSchema] = {}
        self._stores: dict[str, TableStore] = {}
        self.data_dir = data_dir
        if data_dir != ':memory:':
            self._load()

    # -- CatalogInfo Protocol (injected into parser layer) -------------
    def get_table_columns(self, table: str) -> list[str]:
        if table not in self._schemas:
            raise TableNotFoundError(table)
        return self._schemas[table].column_names

    def table_exists(self, table: str) -> bool:
        return table in self._schemas

    # -- DDL -----------------------------------------------------------
    def create_table(self, schema: TableSchema, if_not_exists: bool = False) -> None:
        if schema.name in self._schemas:
            if if_not_exists:
                return
            raise DuplicateError(f"table '{schema.name}' already exists")
        self._schemas[schema.name] = schema
        self._stores[schema.name] = TableStore(schema)
        self._persist()

    def drop_table(self, name: str, if_exists: bool = False) -> None:
        if name not in self._schemas:
            if if_exists:
                return
            raise TableNotFoundError(name)
        del self._schemas[name]
        del self._stores[name]
        self._persist()

    # -- Queries -------------------------------------------------------
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

    # -- Persistence ---------------------------------------------------
    def _persist(self) -> None:
        if self.data_dir != ':memory:':
            from catalog import persist  # lazy import to break cycle
            persist.save(self._schemas, Path(self.data_dir) / 'catalog.json')

    def _load(self) -> None:
        path = Path(self.data_dir) / 'catalog.json'
        if path.exists():
            from catalog import persist  # lazy import to break cycle
            self._schemas = persist.load(path)
            for name, schema in self._schemas.items():
                self._stores[name] = TableStore(schema)
