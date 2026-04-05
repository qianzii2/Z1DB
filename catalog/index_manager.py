from __future__ import annotations
"""索引管理器 — CREATE INDEX / DROP INDEX / 自动维护。"""
from typing import Any, Dict, List, Optional, Tuple
from utils.errors import DuplicateError, ExecutionError

try:
    from structures.art import AdaptiveRadixTree
    _HAS_ART = True
except ImportError:
    _HAS_ART = False

try:
    from structures.skip_list import SkipList
    _HAS_SKIP = True
except ImportError:
    _HAS_SKIP = False


class IndexInfo:
    """索引元数据。"""
    __slots__ = ('name', 'table', 'columns', 'unique', 'index_obj')

    def __init__(self, name: str, table: str, columns: List[str],
                 unique: bool = False) -> None:
        self.name = name
        self.table = table
        self.columns = columns
        self.unique = unique
        self.index_obj: Any = None  # ART or SkipList


class IndexManager:
    """管理所有索引的创建、删除、维护。"""

    def __init__(self) -> None:
        self._indices: Dict[str, IndexInfo] = {}  # index_name → IndexInfo
        self._table_indices: Dict[str, List[str]] = {}  # table → [idx_names]

    def create_index(self, name: str, table: str, columns: List[str],
                     unique: bool = False,
                     if_not_exists: bool = False) -> None:
        """创建索引。"""
        if name in self._indices:
            if if_not_exists:
                return
            raise DuplicateError(f"index '{name}' already exists")

        info = IndexInfo(name, table, columns, unique)
        # 选择索引结构：单列字符串用ART，其他用SkipList
        if len(columns) == 1 and _HAS_ART:
            info.index_obj = AdaptiveRadixTree()
        elif _HAS_SKIP:
            info.index_obj = SkipList()
        else:
            info.index_obj = {}  # dict回退

        self._indices[name] = info
        self._table_indices.setdefault(table, []).append(name)

    def drop_index(self, name: str, if_exists: bool = False) -> None:
        """删除索引。"""
        if name not in self._indices:
            if if_exists:
                return
            raise ExecutionError(f"index '{name}' not found")
        info = self._indices[name]
        if info.table in self._table_indices:
            self._table_indices[info.table] = [
                n for n in self._table_indices[info.table] if n != name]
        del self._indices[name]

    def build_index(self, name: str, store: Any, catalog: Any) -> int:
        """从表数据构建索引。返回索引行数。"""
        if name not in self._indices:
            raise ExecutionError(f"index '{name}' not found")
        info = self._indices[name]
        schema = catalog.get_table(info.table)
        col_indices = []
        for cn in info.columns:
            for i, c in enumerate(schema.columns):
                if c.name == cn:
                    col_indices.append(i)
                    break

        all_rows = store.read_all_rows()
        count = 0
        for row_id, row in enumerate(all_rows):
            key_vals = tuple(row[ci] for ci in col_indices)
            key = key_vals[0] if len(key_vals) == 1 else key_vals
            self._insert_into_index(info, key, row_id)
            count += 1
        return count

    def insert_row(self, table: str, row: list, row_id: int,
                   schema: Any) -> None:
        """INSERT时维护索引。"""
        for idx_name in self._table_indices.get(table, []):
            info = self._indices[idx_name]
            col_indices = []
            for cn in info.columns:
                for i, c in enumerate(schema.columns):
                    if c.name == cn:
                        col_indices.append(i)
                        break
            key_vals = tuple(row[ci] for ci in col_indices)
            key = key_vals[0] if len(key_vals) == 1 else key_vals
            self._insert_into_index(info, key, row_id)

    def get_index_for_column(self, table: str,
                             column: str) -> Optional[Any]:
        """获取指定表列的索引对象。"""
        for idx_name in self._table_indices.get(table, []):
            info = self._indices[idx_name]
            if len(info.columns) == 1 and info.columns[0] == column:
                return info.index_obj
        return None

    def get_table_indices(self, table: str) -> List[IndexInfo]:
        return [self._indices[n] for n in self._table_indices.get(table, [])
                if n in self._indices]

    def has_index(self, name: str) -> bool:
        return name in self._indices

    def list_indices(self) -> List[str]:
        return list(self._indices.keys())

    def _insert_into_index(self, info: IndexInfo, key: Any,
                           row_id: int) -> None:
        idx = info.index_obj
        if _HAS_ART and isinstance(idx, AdaptiveRadixTree):
            kb = str(key).encode('utf-8')
            existing = idx.search(kb)
            if existing is not None:
                if isinstance(existing, list):
                    existing.append(row_id)
                    idx.insert(kb, existing)
                else:
                    idx.insert(kb, [existing, row_id])
            else:
                idx.insert(kb, [row_id])
        elif _HAS_SKIP and isinstance(idx, SkipList):
            existing = idx.search(key)
            if existing is not None:
                if isinstance(existing, list):
                    existing.append(row_id)
                else:
                    idx.insert(key, [existing, row_id])
            else:
                idx.insert(key, [row_id])
        elif isinstance(idx, dict):
            idx.setdefault(key, []).append(row_id)

    def invalidate_table(self, table: str) -> None:
        """表数据变更后重建所有索引。（简单实现）"""
        for idx_name in self._table_indices.get(table, []):
            info = self._indices[idx_name]
            if _HAS_ART and isinstance(info.index_obj, AdaptiveRadixTree):
                info.index_obj = AdaptiveRadixTree()
            elif _HAS_SKIP and isinstance(info.index_obj, SkipList):
                info.index_obj = SkipList()
            elif isinstance(info.index_obj, dict):
                info.index_obj = {}
