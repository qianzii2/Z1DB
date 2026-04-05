from __future__ import annotations
"""Merge Worker — background thread that merges Delta into Main.
Paper: Sikka et al., 2012 "SAP HANA"

Periodically:
  1. Freeze current Delta Store
  2. Merge frozen Delta + Main → new Main
  3. Swap in new Main, discard old
  4. Clear frozen Delta"""
import threading
import time
from typing import Any, Callable, Dict, List, Optional


class MergeWorker:
    """Background merge worker. Runs in a daemon thread."""

    def __init__(self, merge_fn: Callable, interval: float = 30.0,
                 threshold: int = 10000) -> None:
        self._merge_fn = merge_fn
        self._interval = interval
        self._threshold = threshold  # Merge when delta has this many rows
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stats = {'merges': 0, 'total_rows_merged': 0}

    def start(self) -> None:
        """Start background merge thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop background merge thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None

    def trigger_merge(self) -> None:
        """Trigger an immediate merge."""
        try:
            rows_merged = self._merge_fn()
            self._stats['merges'] += 1
            self._stats['total_rows_merged'] += rows_merged
        except Exception:
            pass

    def _run(self) -> None:
        """Main loop: check delta size, merge if needed."""
        while self._running:
            try:
                rows_merged = self._merge_fn()
                if rows_merged > 0:
                    self._stats['merges'] += 1
                    self._stats['total_rows_merged'] += rows_merged
            except Exception:
                pass
            # Sleep in small increments for responsive shutdown
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
