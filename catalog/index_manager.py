from __future__ import annotations
"""索引管理器 — CREATE/DROP INDEX + 自动维护。
集成 cuckoo_filter（DELETE）+ ribbon/xor_filter（静态查找）。"""
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

try:
    from structures.cuckoo_filter import CuckooFilter
    _HAS_CUCKOO = True
except ImportError:
    _HAS_CUCKOO = False

try:
    from structures.ribbon_filter import RibbonFilter
    _HAS_RIBBON = True
except ImportError:
    _HAS_RIBBON = False

try:
    from structures.xor_filter import XorFilter
    _HAS_XOR = True
except ImportError:
    _HAS_XOR = False

# 静态过滤器选择阈值
_STATIC_FILTER_THRESHOLD = 1000


class IndexInfo:
    __slots__ = ('name', 'table', 'columns', 'unique',
                 'index_obj', 'filter_obj', 'static_filter')

    def __init__(self, name: str, table: str,
                 columns: List[str],
                 unique: bool = False) -> None:
        self.name = name
        self.table = table
        self.columns = columns
        self.unique = unique
        self.index_obj: Any = None
        self.filter_obj: Any = None  # CuckooFilter（支持删除）
        self.static_filter: Any = None  # Ribbon/XOR（只读，空间最优）


class IndexManager:
    def __init__(self) -> None:
        self._indices: Dict[str, IndexInfo] = {}
        self._table_indices: Dict[str, List[str]] = {}

    def create_index(self, name: str, table: str,
                     columns: List[str],
                     unique: bool = False,
                     if_not_exists: bool = False) -> None:
        if name in self._indices:
            if if_not_exists:
                return
            raise DuplicateError(
                f"index '{name}' already exists")
        info = IndexInfo(name, table, columns, unique)
        if len(columns) == 1 and _HAS_ART:
            info.index_obj = AdaptiveRadixTree()
        elif _HAS_SKIP:
            info.index_obj = SkipList()
        else:
            info.index_obj = {}
        if _HAS_CUCKOO:
            info.filter_obj = CuckooFilter(capacity=4096)
        self._indices[name] = info
        self._table_indices.setdefault(table, []).append(name)

    def drop_index(self, name: str,
                   if_exists: bool = False) -> None:
        if name not in self._indices:
            if if_exists:
                return
            raise ExecutionError(f"index '{name}' not found")
        info = self._indices[name]
        if info.table in self._table_indices:
            self._table_indices[info.table] = [
                n for n in self._table_indices[info.table]
                if n != name]
        del self._indices[name]

    def build_index(self, name: str, store: Any,
                    catalog: Any) -> int:
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
        if _HAS_CUCKOO:
            info.filter_obj = CuckooFilter(
                capacity=max(len(all_rows) * 2, 1024))

        # 收集 int 类型 key 用于静态过滤器
        int_keys: list = []

        for row_id, row in enumerate(all_rows):
            key_vals = tuple(row[ci] for ci in col_indices)
            key = (key_vals[0] if len(key_vals) == 1
                   else key_vals)
            self._insert_into_index(info, key, row_id)
            # 收集 int key
            if isinstance(key, int):
                int_keys.append(key)
            count += 1

        # 构建静态过滤器（Ribbon > XOR > 无）
        if count >= _STATIC_FILTER_THRESHOLD and int_keys:
            if _HAS_RIBBON:
                try:
                    info.static_filter = RibbonFilter(
                        int_keys, fp_bits=8)
                except Exception:
                    if _HAS_XOR:
                        try:
                            info.static_filter = XorFilter(
                                int_keys, fp_bits=8)
                        except Exception:
                            pass
            elif _HAS_XOR:
                try:
                    info.static_filter = XorFilter(
                        int_keys, fp_bits=8)
                except Exception:
                    pass

        return count

    def insert_row(self, table: str, row: list,
                   row_id: int, schema: Any) -> None:
        for idx_name in self._table_indices.get(table, []):
            info = self._indices[idx_name]
            col_indices = []
            for cn in info.columns:
                for i, c in enumerate(schema.columns):
                    if c.name == cn:
                        col_indices.append(i)
                        break
            key_vals = tuple(row[ci] for ci in col_indices)
            key = (key_vals[0] if len(key_vals) == 1
                   else key_vals)
            self._insert_into_index(info, key, row_id)
            # 插入后静态过滤器失效，标记重建
            info.static_filter = None

    def delete_row(self, table: str, row: list,
                   schema: Any) -> None:
        for idx_name in self._table_indices.get(table, []):
            info = self._indices[idx_name]
            col_indices = []
            for cn in info.columns:
                for i, c in enumerate(schema.columns):
                    if c.name == cn:
                        col_indices.append(i)
                        break
            key_vals = tuple(row[ci] for ci in col_indices)
            key = (key_vals[0] if len(key_vals) == 1
                   else key_vals)
            if info.filter_obj and _HAS_CUCKOO:
                info.filter_obj.delete(
                    str(key).encode('utf-8'))

    def might_contain(self, table: str, column: str,
                      value: Any) -> bool:
        """三级过滤：静态过滤器 → CuckooFilter → True。
        False = 一定不在索引中。True = 可能在。"""
        for idx_name in self._table_indices.get(table, []):
            info = self._indices[idx_name]
            if (len(info.columns) == 1
                    and info.columns[0] == column):
                # 第一级：静态过滤器（空间最优）
                if (info.static_filter is not None
                        and isinstance(value, int)):
                    if not info.static_filter.contains(value):
                        return False
                # 第二级：CuckooFilter
                if info.filter_obj and _HAS_CUCKOO:
                    return info.filter_obj.contains(
                        str(value).encode('utf-8'))
                return True
        return True

    def get_index_for_column(self, table: str,
                             column: str) -> Optional[Any]:
        for idx_name in self._table_indices.get(table, []):
            info = self._indices[idx_name]
            if (len(info.columns) == 1
                    and info.columns[0] == column):
                return info.index_obj
        return None

    def get_filter_for_column(self, table: str,
                              column: str) -> Optional[Any]:
        for idx_name in self._table_indices.get(table, []):
            info = self._indices[idx_name]
            if (len(info.columns) == 1
                    and info.columns[0] == column):
                return info.filter_obj
        return None

    def get_table_indices(self, table: str) -> List[IndexInfo]:
        return [self._indices[n]
                for n in self._table_indices.get(table, [])
                if n in self._indices]

    def has_index(self, name: str) -> bool:
        return name in self._indices

    def list_indices(self) -> List[str]:
        return list(self._indices.keys())

    def _insert_into_index(self, info: IndexInfo,
                           key: Any, row_id: int) -> None:
        idx = info.index_obj
        if info.filter_obj and _HAS_CUCKOO:
            info.filter_obj.add(str(key).encode('utf-8'))

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
        for idx_name in self._table_indices.get(table, []):
            info = self._indices[idx_name]
            if _HAS_ART and isinstance(
                    info.index_obj, AdaptiveRadixTree):
                info.index_obj = AdaptiveRadixTree()
            elif _HAS_SKIP and isinstance(
                    info.index_obj, SkipList):
                info.index_obj = SkipList()
            elif isinstance(info.index_obj, dict):
                info.index_obj = {}
            if _HAS_CUCKOO:
                info.filter_obj = CuckooFilter(capacity=4096)
            info.static_filter = None
