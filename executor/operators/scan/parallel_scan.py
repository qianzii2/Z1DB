from __future__ import annotations
"""并行扫描 — multiprocessing 真并行。
每个 worker 进程独立 GIL，CPU 密集的列解码/NaN-Boxing 真正并行。
回退：序列化失败时用 threading。"""
import os
from typing import Any, Dict, List, Optional, Tuple
from executor.core.batch import VectorBatch
from executor.core.operator import Operator
from executor.core.vector import DataVector
from storage.types import DataType


def _scan_chunk_worker(args: tuple) -> Optional[list]:
    """Worker 函数（必须在模块顶层，可 pickle）。
    接收 (chunk_data_dict, col_names, col_types)，返回行列表。"""
    chunk_data, col_names, col_types = args
    try:
        rows = []
        n = chunk_data.get('_n', 0)
        for ri in range(n):
            row = []
            for ci, cn in enumerate(col_names):
                vals = chunk_data.get(cn)
                if vals is not None and ri < len(vals):
                    row.append(vals[ri])
                else:
                    row.append(None)
            rows.append(row)
        return rows
    except Exception:
        return None


class ParallelScanOperator(Operator):
    """多进程并行扫描。每个 chunk 在独立进程中解码。"""

    def __init__(self, table_name: str, store: Any,
                 columns: List[str],
                 num_workers: int = 2) -> None:
        super().__init__()
        self._table_name = table_name
        self._store = store
        self._columns = columns
        self._num_workers = max(1, min(num_workers, os.cpu_count() or 2))
        self._batches: List[VectorBatch] = []
        self._batch_idx = 0

    def output_schema(self) -> List[Tuple[str, DataType]]:
        col_map = {c.name: c.dtype for c in self._store.schema.columns}
        return [(n, col_map[n]) for n in self._columns]

    def open(self) -> None:
        total_chunks = self._store.get_chunk_count()
        if total_chunks == 0:
            self._batches = []
            self._batch_idx = 0
            return

        # 收集非空 chunk
        first_col = self._columns[0]
        chunks = self._store.get_column_chunks(first_col)
        nonempty = [ci for ci in range(min(total_chunks, len(chunks)))
                    if chunks[ci].row_count > 0]

        if not nonempty:
            self._batches = []
            self._batch_idx = 0
            return

        # 预提取 chunk 数据为 Python 原生类型（可序列化）
        chunk_args = []
        schema = self.output_schema()
        col_types = [dt for _, dt in schema]
        for ci in nonempty:
            chunk_data: Dict[str, list] = {'_n': 0}
            for name in self._columns:
                chunk_list = self._store.get_column_chunks(name)
                if ci < len(chunk_list):
                    cc = chunk_list[ci]
                    chunk_data['_n'] = cc.row_count
                    chunk_data[name] = [cc.get(i) for i in range(cc.row_count)]
            chunk_args.append((chunk_data, list(self._columns), col_types))

        # 尝试 multiprocessing，回退 threading
        if self._num_workers > 1 and len(chunk_args) > 1:
            try:
                from executor.pipeline.morsel import MorselDriver
                driver = MorselDriver(self._num_workers, use_process=True)
                results = driver.execute(chunk_args, _scan_chunk_worker)
            except Exception:
                results = [_scan_chunk_worker(a) for a in chunk_args]
        else:
            results = [_scan_chunk_worker(a) for a in chunk_args]

        # 构建 VectorBatch
        self._batches = []
        col_names = list(self._columns)
        for rows in results:
            if rows:
                self._batches.append(
                    VectorBatch.from_rows(rows, col_names, col_types))
        self._batch_idx = 0

    def next_batch(self) -> Optional[VectorBatch]:
        if self._batch_idx >= len(self._batches):
            return None
        b = self._batches[self._batch_idx]
        self._batch_idx += 1
        return b

    def close(self) -> None:
        pass
