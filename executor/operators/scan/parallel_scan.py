from __future__ import annotations
"""并行扫描 — 多线程chunk扫描。"""
import threading
from typing import Any, Dict, List, Optional, Tuple
from queue import Queue
from executor.core.batch import VectorBatch
from executor.core.operator import Operator
from executor.core.vector import DataVector
from storage.table_store import TableStore
from storage.types import DataType


class ParallelScanOperator(Operator):
    """多线程并行扫描chunk。"""

    def __init__(self, table_name: str, store: TableStore,
                 columns: List[str], num_workers: int = 2) -> None:
        super().__init__()
        self._table_name = table_name
        self._store = store
        self._columns = columns
        self._num_workers = max(1, num_workers)
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

        # 过滤空chunk
        nonempty: List[int] = []
        first_col = self._columns[0]
        chunks = self._store.get_column_chunks(first_col)
        for ci in range(total_chunks):
            if ci < len(chunks) and chunks[ci].row_count > 0:
                nonempty.append(ci)

        if not nonempty:
            self._batches = []
            self._batch_idx = 0
            return

        result_queue: Queue = Queue()

        def worker(chunk_indices: List[int]) -> None:
            for ci in chunk_indices:
                try:
                    cols: Dict[str, DataVector] = {}
                    for name in self._columns:
                        chunk_list = self._store.get_column_chunks(name)
                        if ci < len(chunk_list):
                            cols[name] = DataVector.from_column_chunk(
                                chunk_list[ci])
                    if cols:
                        batch = VectorBatch(
                            columns=cols,
                            _column_order=list(self._columns))
                        result_queue.put((ci, batch))
                except Exception:
                    pass

        # 轮询分配chunk给worker
        worker_chunks: List[List[int]] = [
            [] for _ in range(self._num_workers)]
        for i, ci in enumerate(nonempty):
            worker_chunks[i % self._num_workers].append(ci)

        threads = []
        for wc in worker_chunks:
            if wc:
                t = threading.Thread(target=worker, args=(wc,))
                threads.append(t)
                t.start()

        for t in threads:
            t.join()

        # 按chunk顺序收集结果
        collected: Dict[int, VectorBatch] = {}
        while not result_queue.empty():
            ci, batch = result_queue.get()
            collected[ci] = batch

        self._batches = [collected[i] for i in sorted(collected.keys())]
        self._batch_idx = 0

    def next_batch(self) -> Optional[VectorBatch]:
        if self._batch_idx >= len(self._batches):
            return None
        b = self._batches[self._batch_idx]
        self._batch_idx += 1
        return b

    def close(self) -> None:
        pass
