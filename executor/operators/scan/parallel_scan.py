from __future__ import annotations
"""Parallel scan — threading-based multi-worker chunk scanning."""
import threading
from typing import Any, Dict, List, Optional, Tuple
from queue import Queue
from executor.core.batch import VectorBatch
from executor.core.operator import Operator
from executor.core.vector import DataVector
from storage.table_store import TableStore
from storage.types import DataType


class ParallelScanOperator(Operator):
    """Scans chunks in parallel using threads. mmap releases GIL."""

    def __init__(self, table_name: str, store: TableStore,
                 columns: List[str], num_workers: int = 2) -> None:
        super().__init__()
        self._table_name = table_name
        self._store = store
        self._columns = columns
        self._num_workers = num_workers
        self._queue: Queue = Queue()
        self._batches: List[VectorBatch] = []
        self._batch_idx = 0

    def output_schema(self) -> List[Tuple[str, DataType]]:
        col_map = {c.name: c.dtype for c in self._store.schema.columns}
        return [(n, col_map[n]) for n in self._columns]

    def open(self) -> None:
        total_chunks = self._store.get_chunk_count()
        if total_chunks == 0:
            self._batches = []; self._batch_idx = 0; return

        # Assign chunks to workers
        result_queue: Queue = Queue()
        threads = []

        def worker(chunk_indices: List[int]) -> None:
            for ci in chunk_indices:
                first_col = self._columns[0]
                chunks = self._store.get_column_chunks(first_col)
                if chunks[ci].row_count == 0: continue
                cols: Dict[str, DataVector] = {}
                for name in self._columns:
                    chunk_list = self._store.get_column_chunks(name)
                    cols[name] = DataVector.from_column_chunk(chunk_list[ci])
                batch = VectorBatch(columns=cols, _column_order=list(self._columns))
                result_queue.put((ci, batch))

        # Distribute chunks round-robin
        worker_chunks: List[List[int]] = [[] for _ in range(self._num_workers)]
        for i in range(total_chunks):
            worker_chunks[i % self._num_workers].append(i)

        for wc in worker_chunks:
            if wc:
                t = threading.Thread(target=worker, args=(wc,))
                threads.append(t)
                t.start()

        for t in threads:
            t.join()

        # Collect results in chunk order
        collected: Dict[int, VectorBatch] = {}
        while not result_queue.empty():
            ci, batch = result_queue.get()
            collected[ci] = batch

        self._batches = [collected[i] for i in sorted(collected.keys())]
        self._batch_idx = 0

    def next_batch(self) -> Optional[VectorBatch]:
        if self._batch_idx >= len(self._batches): return None
        b = self._batches[self._batch_idx]
        self._batch_idx += 1
        return b

    def close(self) -> None: pass
