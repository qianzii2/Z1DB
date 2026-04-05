from __future__ import annotations
"""内存列式存储 — 单表的列式chunk存储。chunk满时触发压缩。"""
from typing import TYPE_CHECKING, List, Set
from metal.config import CHUNK_SIZE
from storage.column_chunk import ColumnChunk
from utils.errors import ColumnNotFoundError

if TYPE_CHECKING:
    from catalog.catalog import TableSchema


class TableStore:
    def __init__(self, schema: TableSchema) -> None:
        self.schema = schema
        self._chunks: dict[str, list[ColumnChunk]] = {}
        self.row_count = 0
        for col in schema.columns:
            self._chunks[col.name] = [ColumnChunk(col.dtype)]

    def append_row(self, row: list) -> None:
        first_col = self.schema.columns[0].name
        current_chunk = self._chunks[first_col][-1]

        # chunk满时：新建chunk，并对旧chunk触发压缩
        if current_chunk.row_count >= CHUNK_SIZE:
            for col in self.schema.columns:
                old_chunk = self._chunks[col.name][-1]
                # 触发字典编码构建（如果尚未构建）
                old_chunk.build_dict_encoding()
                # 新建下一个chunk
                self._chunks[col.name].append(ColumnChunk(col.dtype))

        for i, col in enumerate(self.schema.columns):
            self._chunks[col.name][-1].append(row[i])
        self.row_count += 1

    def append_rows(self, rows: list[list]) -> int:
        for row in rows:
            self.append_row(row)
        return len(rows)

    def get_chunk_count(self) -> int:
        if not self.schema.columns:
            return 0
        return len(self._chunks[self.schema.columns[0].name])

    def get_column_chunks(self, column_name: str) -> list[ColumnChunk]:
        if column_name not in self._chunks:
            raise ColumnNotFoundError(column_name)
        return self._chunks[column_name]

    def truncate(self) -> None:
        for col in self.schema.columns:
            self._chunks[col.name] = [ColumnChunk(col.dtype)]
        self.row_count = 0

    def read_all_rows(self) -> List[list]:
        """读取所有行（按schema列顺序）。"""
        rows: List[list] = []
        chunk_count = self.get_chunk_count()
        for chunk_idx in range(chunk_count):
            first = self._chunks[self.schema.columns[0].name][chunk_idx]
            for row_idx in range(first.row_count):
                row = []
                for col in self.schema.columns:
                    row.append(
                        self._chunks[col.name][chunk_idx].get(row_idx))
                rows.append(row)
        return rows

    def read_rows_by_indices(self, indices: List[int]) -> List[list]:
        """按全局行索引读取指定行（避免read_all_rows全量读取）。"""
        if not indices:
            return []
        # 预计算每个索引对应的chunk和行内偏移
        result = []
        chunk_count = self.get_chunk_count()
        chunk_starts: List[int] = []
        offset = 0
        for ci in range(chunk_count):
            chunk_starts.append(offset)
            first = self._chunks[self.schema.columns[0].name][ci]
            offset += first.row_count

        for idx in sorted(indices):
            if idx < 0 or idx >= self.row_count:
                continue
            # 二分查找chunk
            ci = self._find_chunk(idx, chunk_starts)
            ri = idx - chunk_starts[ci]
            row = []
            for col in self.schema.columns:
                row.append(self._chunks[col.name][ci].get(ri))
            result.append(row)
        return result

    def _find_chunk(self, global_idx: int,
                    chunk_starts: List[int]) -> int:
        """二分查找全局行索引所在的chunk。"""
        lo, hi = 0, len(chunk_starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if chunk_starts[mid] <= global_idx:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def delete_rows(self, indices_to_delete: Set[int]) -> int:
        """按全局行索引删除行。"""
        if not indices_to_delete:
            return 0
        all_rows = self.read_all_rows()
        remaining = [r for i, r in enumerate(all_rows)
                     if i not in indices_to_delete]
        deleted = len(all_rows) - len(remaining)
        self.truncate()
        for r in remaining:
            self.append_row(r)
        return deleted

    def update_rows(self, indices: Set[int], col_idx: int,
                    new_values: dict) -> int:
        """更新指定行的指定列值。"""
        all_rows = self.read_all_rows()
        for idx, val in new_values.items():
            if 0 <= idx < len(all_rows):
                all_rows[idx][col_idx] = val
        self.truncate()
        for r in all_rows:
            self.append_row(r)
        return len(new_values)
