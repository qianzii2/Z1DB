from __future__ import annotations
"""LSM 存储引擎 — MemTable → SSTable → Compaction。
[B05] 完全通过 tombstone 实现删除，移除独立 _deleted 集合。
[P04] read_all_rows 带版本号缓存。
集成 BufferPool 缓存热 SSTable 扫描结果。"""
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
    写入 → MemTable → 刷盘 SSTable → 后台 Compaction。"""

    def __init__(self, schema: Any, data_dir: str) -> None:
        if not _HAS_LSM:
            raise ImportError("LSM 组件不可用")
        self.schema = schema
        self._data_dir = os.path.join(
            data_dir, f'lsm_{schema.name}')
        os.makedirs(self._data_dir, exist_ok=True)

        self._memtable = MemTable(capacity=CHUNK_SIZE)
        self._manifest = Manifest(self._data_dir)
        self._compactor = Compactor(self._data_dir)
        self._next_key = 0
        self._row_count = 0
        # [B05] 移除 self._deleted — 完全依赖 tombstone

        # [P04] read_all_rows 缓存
        self._all_rows_cache: Optional[List[list]] = None
        self._all_rows_version: int = 0
        self._version: int = 0

        # 列缓存（scan 算子用）
        self._chunk_cache: Optional[Dict[str, List[ColumnChunk]]] = None
        self._cache_row_count = 0

        # BufferPool 缓存 SSTable 扫描结果
        self._buffer_pool: Optional[BufferPool] = None
        if _HAS_BUFFER_POOL:
            self._buffer_pool = BufferPool(max_pages=256)

        self._recover_state()

    def _recover_state(self) -> None:
        """从 SSTable + MemTable 恢复状态。"""
        max_key = -1
        row_count = 0
        try:
            from storage.lsm.merge_iterator import MergeIterator
            sources = []
            for st in self._manifest.all_sstables():
                sources.append(st.scan())
            sources.append(iter(self._memtable.scan()))
            for key, row in MergeIterator(sources):
                row_count += 1
                try:
                    k = int(key)
                    if k > max_key:
                        max_key = k
                except (ValueError, TypeError):
                    pass
        except ImportError:
            # 回退路径：按 source 顺序遍历，后来的覆盖先前的
            # SSTable 按层级排列（L0 最新在后），MemTable 最新
            merged = {}
            all_sstables = self._manifest.all_sstables()
            for si, st in enumerate(all_sstables):
                for key, row in st.scan():
                    if row is None:
                        # tombstone：标记删除
                        merged[key] = (None, si)
                    else:
                        merged[key] = (row, si)
            # MemTable 是最新的源
            mem_source_id = len(all_sstables)
            for key, row in self._memtable.scan():
                if row is None:
                    merged[key] = (None, mem_source_id)
                else:
                    merged[key] = (row, mem_source_id)
            # 统计存活行（排除 tombstone）
            row_count = 0
            for key, (row, _) in merged.items():
                if row is not None:
                    row_count += 1
                    try:
                        k = int(key)
                        if k > max_key:
                            max_key = k
                    except (ValueError, TypeError):
                        pass
        self._next_key = max_key + 1 if max_key >= 0 else 0
        self._row_count = row_count

    @property
    def row_count(self) -> int:
        return self._row_count

    # ═══ 写入 ═══

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
        self._invalidate_cache()
        if self._buffer_pool:
            self._buffer_pool = BufferPool(max_pages=256)
    # ═══ 读取 [P04] ═══

    def read_all_rows(self) -> List[list]:
        """使用 MergeIterator 读取所有活跃行。带版本号缓存。"""
        if (self._all_rows_cache is not None
                and self._all_rows_version == self._version):
            return [list(r) for r in self._all_rows_cache]

        try:
            from storage.lsm.merge_iterator import MergeIterator
            sources = []
            for st in self._manifest.all_sstables():
                sources.append(self._scan_sstable_cached(st))
            sources.append(iter(self._memtable.scan()))
            rows = [row for _, row in MergeIterator(sources)]
        except ImportError:
            merged = {}
            for st in self._manifest.all_sstables():
                for key, row in self._scan_sstable_cached(st):
                    if row is None:
                        merged.pop(key, None)
                    else:
                        merged[key] = row
            for key, row in self._memtable.scan():
                if row is None:
                    merged.pop(key, None)
                else:
                    merged[key] = row
            rows = [row for _, row in sorted(merged.items())]

        self._all_rows_cache = rows
        self._all_rows_version = self._version
        return [list(r) for r in rows]

    def _scan_sstable_cached(self, st):
        """SSTable 扫描（通过 BufferPool 缓存结果）。"""
        if self._buffer_pool is None:
            yield from st.scan()
            return
        page_id = f"sst:{st.path}"
        cached = self._buffer_pool.get(page_id)
        if cached is not None:
            yield from cached
            return
        entries = list(st.scan())
        self._buffer_pool.put(page_id, entries)
        yield from entries

    def read_rows_by_indices(self,
                             indices: List[int]) -> List[list]:
        if not indices:
            return []
        all_rows = self.read_all_rows()
        return [all_rows[idx] for idx in sorted(indices)
                if 0 <= idx < len(all_rows)]

    # ═══ 删除 [B05] ═══

    def delete_rows(self, indices_to_delete: Set[int]) -> int:
        """删除指定逻辑行。写入 tombstone 到 MemTable。
        [B05] 完全通过 tombstone，不维护独立 _deleted 集合。"""
        if not indices_to_delete:
            return 0
        active_keys = self._get_active_keys()
        count = 0
        for idx in sorted(indices_to_delete):
            if 0 <= idx < len(active_keys):
                key = active_keys[idx]
                self._memtable.put(key, None)  # tombstone
                count += 1
        self._row_count = max(0, self._row_count - count)
        self._invalidate_cache()
        if self._memtable.is_full:
            self._flush()
        return count

    def update_rows(self, indices: Set[int], col_idx: int,
                    new_values: dict) -> int:
        """更新指定行。写入新行 + tombstone 旧行。"""
        if not new_values:
            return 0
        all_rows = self.read_all_rows()
        active_keys = self._get_active_keys()
        count = 0
        for logical_idx, new_val in new_values.items():
            if (0 <= logical_idx < len(all_rows)
                    and logical_idx < len(active_keys)):
                old_key = active_keys[logical_idx]
                self._memtable.put(old_key, None)  # tombstone
                self._row_count -= 1
                old_row = list(all_rows[logical_idx])
                old_row[col_idx] = new_val
                self.append_row(old_row)
                count += 1
        self._invalidate_cache()
        return count

    # ═══ TableStore 兼容接口 ═══

    def get_chunk_count(self) -> int:
        self._ensure_chunk_cache()
        if not self.schema.columns:
            return 0
        first = self.schema.columns[0].name
        return len(self._chunk_cache.get(first, []))

    def get_column_chunks(self,
                          column_name: str) -> List[ColumnChunk]:
        self._ensure_chunk_cache()
        if column_name not in self._chunk_cache:
            raise ColumnNotFoundError(column_name)
        return self._chunk_cache[column_name]

    def _ensure_chunk_cache(self) -> None:
        if (self._chunk_cache is not None
                and self._cache_row_count == self._row_count):
            return
        self._rebuild_chunk_cache()

    def _rebuild_chunk_cache(self) -> None:
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
        self._version += 1
        self._all_rows_cache = None
        self._chunk_cache = None
        if self._buffer_pool:
            for st in self._manifest.all_sstables():
                self._buffer_pool.invalidate(f"sst:{st.path}")

    def _get_active_keys(self) -> List[Any]:
        """获取所有活跃行的 key（有序）。"""
        try:
            from storage.lsm.merge_iterator import MergeIterator
            sources = []
            for st in self._manifest.all_sstables():
                sources.append(self._scan_sstable_cached(st))
            sources.append(iter(self._memtable.scan()))
            return [key for key, _ in MergeIterator(sources)]
        except ImportError:
            merged = {}
            for st in self._manifest.all_sstables():
                for key, row in self._scan_sstable_cached(st):
                    if row is None:
                        merged.pop(key, None)
                    else:
                        merged[key] = True
            for key, row in self._memtable.scan():
                if row is None:
                    merged.pop(key, None)
                else:
                    merged[key] = True
            return sorted(
                k for k, alive in merged.items() if alive)

    # ═══ LSM 内部操作 ═══

    def _flush(self) -> None:
        """将 MemTable 刷盘为 SSTable。"""
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
        """L0 → L1 合并。"""
        try:
            l0 = self._manifest.get_sstables(0)
            if len(l0) < 2:
                return
            old_paths = [st.path for st in l0]
            output = self._manifest.next_sstable_path(level=1)
            self._compactor.compact(
                l0, output, self.schema.column_names)
            self._manifest.add_sstable(1, output)
            for path in old_paths:
                self._manifest.remove_sstable(0, path)
            for st in l0:
                if self._buffer_pool:
                    self._buffer_pool.invalidate(
                        f"sst:{st.path}")
                st.delete_files()
        except Exception:
            pass

    def flush(self) -> None:
        if self._memtable.size > 0:
            self._flush()

    def close(self) -> None:
        self.flush()
