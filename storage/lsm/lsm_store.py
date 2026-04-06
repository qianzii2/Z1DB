from __future__ import annotations
"""LSM 存储 — MemTable → SSTable → Compaction。
[集成] BufferPool 缓存热 SSTable 扫描结果，避免重复 I/O。"""
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

from metal.config import CHUNK_SIZE
from storage.column_chunk import ColumnChunk
from utils.errors import ColumnNotFoundError

if TYPE_CHECKING:
    from catalog.catalog import TableSchema

try:
    from storage.lsm.memtable import MemTable
    from storage.lsm.sstable import SSTableWriter, SSTableReader
    from storage.lsm.manifest import Manifest
    from storage.lsm.compaction import Compactor
    _HAS_LSM = True
except ImportError:
    _HAS_LSM = False

try:
    from storage.io.buffer_pool import BufferPool
    _HAS_BUFFER_POOL = True
except ImportError:
    _HAS_BUFFER_POOL = False


class LSMStore:
    """LSM-Tree 存储，兼容 TableStore 接口。
    写入 → MemTable → 刷盘 SSTable → 后台 Compaction。
    [集成] 读取时通过 BufferPool 缓存 SSTable 扫描结果。"""

    def __init__(self, schema: Any, data_dir: str) -> None:
        if not _HAS_LSM:
            raise ImportError("LSM 组件不可用")
        self.schema = schema
        self._data_dir = os.path.join(data_dir, f'lsm_{schema.name}')
        os.makedirs(self._data_dir, exist_ok=True)

        self._memtable = MemTable(capacity=CHUNK_SIZE)
        self._manifest = Manifest(self._data_dir)
        self._compactor = Compactor(self._data_dir)
        self._next_key = 0
        self._row_count = 0
        self._deleted: Set[int] = set()

        # 列缓存（读取时构建，写入时失效）
        self._chunk_cache: Optional[Dict[str, List[ColumnChunk]]] = None
        self._cache_row_count = 0

        # [集成] BufferPool 缓存 SSTable 扫描结果
        self._buffer_pool: Optional[BufferPool] = None
        if _HAS_BUFFER_POOL:
            self._buffer_pool = BufferPool(max_pages=256)

        self._recover_state()

    def _recover_state(self) -> None:
        """从 SSTable 恢复行数和 next_key（去重计数）。"""
        merged: Dict[Any, bool] = {}
        for st in self._manifest.all_sstables():
            for key, row in self._scan_sstable(st):
                if row is None:
                    merged[key] = False
                    self._deleted.add(key)
                else:
                    if key not in self._deleted:
                        merged[key] = True
                try:
                    k = int(key)
                    if k >= self._next_key:
                        self._next_key = k + 1
                except (ValueError, TypeError):
                    pass
        self._row_count = sum(1 for alive in merged.values() if alive)

    @property
    def row_count(self) -> int:
        return self._row_count

    def append_row(self, row: list) -> None:
        key = self._next_key
        self._next_key += 1
        self._memtable.put(key, list(row))
        self._row_count += 1
        self._invalidate_cache()
        if self._memtable.is_full:
            self._flush()

    def append_rows(self, rows: List[list]) -> int:
        for row in rows:
            self.append_row(row)
        return len(rows)

    def truncate(self) -> None:
        self._memtable.clear()
        for st in self._manifest.all_sstables():
            st.delete_files()
        self._manifest = Manifest(self._data_dir)
        self._row_count = 0
        self._next_key = 0
        self._deleted.clear()
        self._invalidate_cache()
        if self._buffer_pool:
            self._buffer_pool = BufferPool(max_pages=256)

    def read_all_rows(self) -> List[list]:
        """读取所有活跃行（合并 MemTable + SSTable）。"""
        merged: Dict[Any, list] = {}
        for st in self._manifest.all_sstables():
            for key, row in self._scan_sstable(st):
                if row is None:
                    merged.pop(key, None)
                else:
                    merged[key] = row
        for key, row in self._memtable.scan():
            if row is None:
                merged.pop(key, None)
            else:
                merged[key] = row
        for dk in self._deleted:
            merged.pop(dk, None)
        sorted_items = sorted(merged.items(), key=lambda x: x[0])
        return [row for _, row in sorted_items]

    def read_rows_by_indices(self, indices: List[int]) -> List[list]:
        if not indices:
            return []
        all_rows = self.read_all_rows()
        return [all_rows[idx] for idx in sorted(indices)
                if 0 <= idx < len(all_rows)]

    def delete_rows(self, indices_to_delete: Set[int]) -> int:
        if not indices_to_delete:
            return 0
        active_keys = self._get_active_keys()
        count = 0
        for idx in sorted(indices_to_delete):
            if 0 <= idx < len(active_keys):
                key = active_keys[idx]
                self._memtable.delete(key)
                self._deleted.add(key)
                count += 1
        self._row_count = max(0, self._row_count - count)
        self._invalidate_cache()
        if self._memtable.is_full:
            self._flush()
        return count

    def update_rows(self, indices: Set[int], col_idx: int,
                    new_values: dict) -> int:
        all_rows = self.read_all_rows()
        active_keys = self._get_active_keys()
        count = 0
        for idx, val in new_values.items():
            if 0 <= idx < len(all_rows) and idx < len(active_keys):
                old_row = list(all_rows[idx])
                old_row[col_idx] = val
                old_key = active_keys[idx]
                self._memtable.delete(old_key)
                self._deleted.add(old_key)
                self._row_count -= 1
                self.append_row(old_row)
                count += 1
        self._invalidate_cache()
        return count

    # ═══ [集成] BufferPool 缓存 SSTable 扫描 ═══

    def _scan_sstable(self, st: 'SSTableReader'):
        """通过 BufferPool 缓存 SSTable 扫描结果。"""
        if self._buffer_pool is None:
            yield from st.scan()
            return
        page_id = f"sst:{st.path}"
        cached = self._buffer_pool.get(page_id)
        if cached is not None:
            yield from cached
            return
        # 缓存未命中：扫描并缓存
        entries = list(st.scan())
        self._buffer_pool.put(page_id, entries)
        yield from entries

    # ═══ TableStore 兼容接口 ═══

    def get_chunk_count(self) -> int:
        self._ensure_cache()
        if not self.schema.columns:
            return 0
        first = self.schema.columns[0].name
        return len(self._chunk_cache.get(first, []))

    def get_column_chunks(self, column_name: str) -> List[ColumnChunk]:
        self._ensure_cache()
        if column_name not in self._chunk_cache:
            raise ColumnNotFoundError(column_name)
        return self._chunk_cache[column_name]

    def _ensure_cache(self) -> None:
        if (self._chunk_cache is not None
                and self._cache_row_count == self._row_count):
            return
        self._rebuild_cache()

    def _rebuild_cache(self) -> None:
        all_rows = self.read_all_rows()
        n = len(all_rows)
        self._chunk_cache = {}
        for col in self.schema.columns:
            chunks: List[ColumnChunk] = []
            col_idx = next(
                (i for i, c in enumerate(self.schema.columns)
                 if c.name == col.name), 0)
            for start in range(0, max(n, 1), CHUNK_SIZE):
                chunk = ColumnChunk(col.dtype)
                end = min(start + CHUNK_SIZE, n)
                for ri in range(start, end):
                    if ri < n:
                        chunk.append(all_rows[ri][col_idx])
                chunks.append(chunk)
            if not chunks:
                chunks.append(ColumnChunk(col.dtype))
            self._chunk_cache[col.name] = chunks
        self._cache_row_count = self._row_count

    def _invalidate_cache(self) -> None:
        self._chunk_cache = None
        # 写入后失效 BufferPool（数据已变）
        if self._buffer_pool:
            for st in self._manifest.all_sstables():
                self._buffer_pool.invalidate(f"sst:{st.path}")

    def _get_active_keys(self) -> List[Any]:
        merged: Dict[Any, bool] = {}
        for st in self._manifest.all_sstables():
            for key, row in self._scan_sstable(st):
                if row is None:
                    merged.pop(key, None)
                else:
                    if key not in self._deleted:
                        merged[key] = True
        for key, row in self._memtable.scan():
            if row is None:
                merged.pop(key, None)
            else:
                merged[key] = True
        return sorted(k for k, alive in merged.items() if alive)

    # ═══ LSM 内部操作 ═══

    def _flush(self) -> None:
        if self._memtable.size == 0:
            return
        self._memtable.freeze()
        path = self._manifest.next_sstable_path(level=0)
        writer = SSTableWriter(path, self.schema.column_names)
        for key, row in self._memtable.scan():
            writer.add(key, row)
        writer.finish()
        self._manifest.add_sstable(0, path)
        self._memtable = MemTable(capacity=CHUNK_SIZE)
        if self._manifest.level_count(0) > 4:
            self._compact()

    def _compact(self) -> None:
        try:
            l0 = self._manifest.get_sstables(0)
            if len(l0) < 2:
                return
            old_paths = [st.path for st in l0]
            output = self._manifest.next_sstable_path(level=1)
            self._compactor.compact(l0, output, self.schema.column_names)
            self._manifest.add_sstable(1, output)
            for path in old_paths:
                self._manifest.remove_sstable(0, path)
            for st in l0:
                # 失效 BufferPool 中的旧 SSTable 缓存
                if self._buffer_pool:
                    self._buffer_pool.invalidate(f"sst:{st.path}")
                st.delete_files()
            self._deleted.clear()
        except Exception:
            pass

    def flush(self) -> None:
        if self._memtable.size > 0:
            self._flush()

    def close(self) -> None:
        self.flush()
