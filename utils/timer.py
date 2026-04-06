from __future__ import annotations
"""高精度计时器上下文管理器。"""
import time


class Timer:
    """用法: with Timer() as t: ...; print(t.elapsed)"""

    def __init__(self) -> None:
        self.elapsed: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> 'Timer':
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        self.elapsed = time.perf_counter() - self._start
