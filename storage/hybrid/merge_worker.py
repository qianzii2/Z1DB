from __future__ import annotations
"""后台合并线程 — 定时将 Delta Store 合并到 Main Store。
论文: Sikka et al., 2012 "SAP HANA"
以 daemon 线程运行，可安全 stop。"""
import threading
import time
from typing import Any, Callable, Optional


class MergeWorker:
    """后台合并工作线程。"""

    def __init__(self, merge_fn: Callable,
                 interval: float = 30.0,
                 threshold: int = 10000) -> None:
        self._merge_fn = merge_fn
        self._interval = interval
        self._threshold = threshold
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stats = {'merges': 0, 'total_rows_merged': 0}

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None

    def trigger_merge(self) -> None:
        """手动触发合并。"""
        try:
            rows_merged = self._merge_fn()
            self._stats['merges'] += 1
            self._stats['total_rows_merged'] += rows_merged
        except Exception:
            pass

    def _run(self) -> None:
        """主循环：定时检查并合并。"""
        while self._running:
            try:
                rows_merged = self._merge_fn()
                if rows_merged > 0:
                    self._stats['merges'] += 1
                    self._stats['total_rows_merged'] += rows_merged
            except Exception:
                pass
            # 分段睡眠，便于快速停止
            for _ in range(int(self._interval * 10)):
                if not self._running:
                    break
                time.sleep(0.1)

    @property
    def stats(self) -> dict:
        return dict(self._stats)

    @property
    def is_running(self) -> bool:
        return self._running
