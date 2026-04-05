from __future__ import annotations
"""Morsel-driven parallelism. Paper: Leis et al., 2014.
Splits data into morsels → N workers consume from shared queue."""
import threading
from typing import Any, Callable, Dict, List, Optional
from queue import Queue
from metal.config import CHUNK_SIZE


class MorselDriver:
    """Distributes morsels of work to thread workers."""

    MORSEL_SIZE = CHUNK_SIZE  # 64K rows per morsel

    def __init__(self, num_workers: int = 2) -> None:
        self._num_workers = max(1, num_workers)

    def execute(self, morsels: List[Any],
                worker_fn: Callable[[Any], Any]) -> List[Any]:
        """Execute worker_fn on each morsel using thread pool.
        Returns collected results."""
        if len(morsels) <= 1 or self._num_workers <= 1:
            return [worker_fn(m) for m in morsels]

        work_queue: Queue = Queue()
        result_queue: Queue = Queue()
        for i, m in enumerate(morsels):
            work_queue.put((i, m))

        def worker() -> None:
            while True:
                try:
                    idx, morsel = work_queue.get_nowait()
                except Exception:
                    break
                result = worker_fn(morsel)
                result_queue.put((idx, result))

        threads = []
        for _ in range(min(self._num_workers, len(morsels))):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

        # Collect results in order
        collected: Dict[int, Any] = {}
        while not result_queue.empty():
            idx, result = result_queue.get()
            collected[idx] = result
        return [collected[i] for i in sorted(collected.keys())]

    @staticmethod
    def split_into_morsels(data: list, morsel_size: int = CHUNK_SIZE) -> List[list]:
        """Split a list into morsel-sized chunks."""
        return [data[i:i+morsel_size] for i in range(0, len(data), morsel_size)]
