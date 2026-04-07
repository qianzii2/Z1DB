from __future__ import annotations

"""Morsel-Driven Parallelism — 数据分片并行执行。
论文: Leis et al., 2014 "Morsel-Driven Parallelism"
将数据分成 morsel（小块），分配给工作线程并行处理。"""
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Any, Callable, List, Optional


class MorselDriver:
    """Morsel 驱动的并行执行器。
    将任务列表分配到多个 worker 并行执行。"""

    def __init__(self, num_workers: int = 2,
                 use_process: bool = False) -> None:
        self._num_workers = max(1, min(
            num_workers, os.cpu_count() or 4))
        self._use_process = use_process

    def execute(self, tasks: list,
                worker_fn: Callable) -> List[Any]:
        """并行执行任务列表。

        Args:
            tasks: 任务参数列表，每个元素传给 worker_fn
            worker_fn: 工作函数，签名 fn(task) -> result

        Returns:
            结果列表，顺序与 tasks 对应
        """
        if not tasks:
            return []
        if len(tasks) == 1 or self._num_workers <= 1:
            return [worker_fn(t) for t in tasks]

        pool_cls = (ProcessPoolExecutor if self._use_process
                    else ThreadPoolExecutor)
        try:
            with pool_cls(max_workers=self._num_workers) as pool:
                futures = [pool.submit(worker_fn, t) for t in tasks]
                results = []
                for f in futures:
                    try:
                        results.append(f.result(timeout=300))
                    except Exception:
                        results.append(None)
                return results
        except Exception:
            # 回退到串行
            return [worker_fn(t) for t in tasks]

    @property
    def num_workers(self) -> int:
        return self._num_workers
