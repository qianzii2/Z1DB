# storage/table_store.py — 完整文件

from __future__ import annotations
"""内存列式存储 — 列式 chunk + Delta tombstone。
[R6 修复] 写入直接进主存储（保证 SeqScan 可见性）。
DeltaStore 仅用于 UPDATE/DELETE 的 tombstone 跟踪。"""
from typing import TYPE_CHECKING, Dict, List, Optional, Set
from metal.config import CHUNK_SIZE
from storage.column_chunk import ColumnChunk
from utils.errors import ColumnNotFoundError

if TYPE_CHECKING:
    from catalog.catalog import TableSchema

try:
    from storage.hybrid.delta_store import DeltaStore
    _HAS_DELTA = True
except ImportError:
    _HAS_DELTA = False


class TableStore:
    def __init__(self, schema: 'TableSchema') -> None:
        self.schema = schema
        self._chunks: Dict[str, List[ColumnChunk]] = {}
        self._row_count_main = 0
        self._delta: Optional[DeltaStore] = None
        self._deleted_global: Set[int] = set()
        for col in schema.columns:
            self._chunks[col.name] = [ColumnChunk(col.dtype)]
        if _HAS_DELTA:
            self._delta = DeltaStore(schema.column_names)

    @property
    def row_count(self) -> int:
        """逻辑行数 = 主存储 - 已删除。"""
        return self._row_count_main - len(self._deleted_global)

    def append_row(self, row: list) -> None:
        """[R6] 直接追加到主存储（保证 SeqScan 立即可见）。"""
        first_col = self.schema.columns[0].name
        current_chunk = self._chunks[first_col][-1]
        if current_chunk.row_count >= CHUNK_SIZE:
            for col in self.schema.columns:
                old_chunk = self._chunks[col.name][-1]
                old_chunk.build_dict_encoding()
                old_chunk.compress()
                self._chunks[col.name].append(ColumnChunk(col.dtype))
        for i, col in enumerate(self.schema.columns):
            val = row[i] if i < len(row) else None
            self._chunks[col.name][-1].append(val)
        self._row_count_main += 1

    def append_rows(self, rows: List[list]) -> int:
        for row in rows:
            self.append_row(row)
        return len(rows)

    def delete_rows(self, indices_to_delete: Set[int]) -> int:
        """按逻辑索引删除行。"""
        if not indices_to_delete:
            return 0
        index_map = self._build_active_index_map()
        count = 0
        for logical_idx in sorted(indices_to_delete):
            if logical_idx >= len(index_map):
                continue
            entry = index_map[logical_idx]
            if entry is None:
                continue
            physical_idx = entry
            if physical_idx not in self._deleted_global:
                self._deleted_global.add(physical_idx)
                count += 1
        if (self._row_count_main > 0
                and len(self._deleted_global) > self._row_count_main // 2):
            self._compact()
        return count

    def update_rows(self, indices: Set[int], col_idx: int,
                    new_values: dict) -> int:
        """按逻辑索引更新。标记旧行删除 + 追加新行。"""
        if not new_values:
            return 0
        index_map = self._build_active_index_map()
        all_active = self.read_all_rows()
        count = 0
        for logical_idx, val in new_values.items():
            if logical_idx < 0 or logical_idx >= len(all_active):
                continue
            old_row = list(all_active[logical_idx])
            old_row[col_idx] = val
            # 标记旧行删除
            if logical_idx < len(index_map) and index_map[logical_idx] is not None:
                self._deleted_global.add(index_map[logical_idx])
            # 追加新行到主存储
            self.append_row(old_row)
            count += 1
        return count

    def _build_active_index_map(self) -> List[Optional[int]]:
        """构建逻辑索引→物理索引映射。
        逻辑索引 = read_all_rows 返回列表的位置。
        物理索引 = 主存储中的全局行号。"""
        result: List[Optional[int]] = []
        global_idx = 0
        chunk_count = self.get_chunk_count()
        for chunk_idx in range(chunk_count):
            first = self._chunks[self.schema.columns[0].name][chunk_idx]
            for row_idx in range(first.row_count):
                if global_idx not in self._deleted_global:
                    result.append(global_idx)
                global_idx += 1
        return result

    def get_chunk_count(self) -> int:
        if not self.schema.columns:
            return 0
        return len(self._chunks[self.schema.columns[0].name])

    def get_column_chunks(self, column_name: str) -> List[ColumnChunk]:
        if column_name not in self._chunks:
            raise ColumnNotFoundError(column_name)
        return self._chunks[column_name]

    def truncate(self) -> None:
        for col in self.schema.columns:
            self._chunks[col.name] = [ColumnChunk(col.dtype)]
        self._row_count_main = 0
        self._deleted_global.clear()
        if _HAS_DELTA and self._delta:
            self._delta.clear()

    def read_all_rows(self) -> List[list]:
        """读取所有活跃行（跳过已删除行）。"""
        rows: List[list] = []
        global_idx = 0
        chunk_count = self.get_chunk_count()
        for chunk_idx in range(chunk_count):
            first = self._chunks[self.schema.columns[0].name][chunk_idx]
            for row_idx in range(first.row_count):
                if global_idx not in self._deleted_global:
                    row = []
                    for col in self.schema.columns:
                        row.append(
                            self._chunks[col.name][chunk_idx].get(row_idx))
                    rows.append(row)
                global_idx += 1
        return rows

    def read_rows_by_indices(self, indices: List[int]) -> List[list]:
        if not indices:
            return []
        all_rows = self.read_all_rows()
        return [all_rows[idx] for idx in sorted(indices)
                if 0 <= idx < len(all_rows)]

    def _compact(self) -> None:
        """合并：重建主存储，清除删除标记。"""
        active = self.read_all_rows()
        self._chunks = {}
        for col in self.schema.columns:
            self._chunks[col.name] = [ColumnChunk(col.dtype)]
        self._row_count_main = 0
        self._deleted_global.clear()
        if self._delta and _HAS_DELTA:
            self._delta.clear()
        for row in active:
            self.append_row(row)
