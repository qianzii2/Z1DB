from __future__ import annotations
"""Morsel 驱动并行执行器。
将工作分成 morsel（数据块），用线程池或进程池并行执行。
线程池：适合 I/O 密集的解码/扫描任务（GIL 下仍有并发效益）。
进程池：适合 CPU 密集的表达式求值（真并行，绕过 GIL）。"""
import os
from concurrent.futures import (
    ThreadPoolExecutor, ProcessPoolExecutor, Future)
from typing import Any, Callable, List, Optional


class MorselDriver:
    """Morsel 驱动器：将任务分发到工作线程/进程。

    用法:
        driver = MorselDriver(num_workers=4)
        results = driver.execute(morsels, process_fn)
    """

    def __init__(self, num_workers: int = 0,
                 use_process: bool = False) -> None:
        if num_workers <= 0:
            num_workers = min(os.cpu_count() or 2, 4)
        self._num_workers = num_workers
        self._use_process = use_process

    @property
    def num_workers(self) -> int:
        return self._num_workers

    def execute(self, morsels: list,
                process_fn: Callable) -> List[Any]:
        """并行执行 process_fn(morsel) 对每个 morsel。
        返回结果列表（与 morsels 同序）。

        参数:
            morsels: 工作单元列表（chunk 索引、数据段等）
            process_fn: 处理函数，签名 process_fn(morsel) → result
        """
        if not morsels:
            return []

        # 单 morsel 或单 worker：直接顺序执行
        if len(morsels) == 1 or self._num_workers <= 1:
            return [process_fn(m) for m in morsels]

        # 选择执行器
        pool_cls = (ProcessPoolExecutor if self._use_process
                    else ThreadPoolExecutor)

        try:
            with pool_cls(
                    max_workers=self._num_workers) as pool:
                futures: List[Future] = [
                    pool.submit(process_fn, m)
                    for m in morsels]
                results = []
                for fut in futures:
                    try:
                        results.append(fut.result(timeout=300))
                    except Exception:
                        results.append(None)
                return results
        except Exception:
            # 并行失败时回退顺序执行
            return [process_fn(m) for m in morsels]

    def execute_chunked(self, total_items: int,
                        chunk_size: int,
                        process_fn: Callable) -> List[Any]:
        """将 [0, total_items) 分成 chunk_size 大小的块并行执行。
        process_fn 签名: process_fn((start, end)) → result"""
        morsels = []
        for start in range(0, total_items, chunk_size):
            end = min(start + chunk_size, total_items)
            morsels.append((start, end))
        return self.execute(morsels, process_fn)
