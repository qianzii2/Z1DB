from __future__ import annotations
"""LRU-K buffer pool — keeps hot pages in memory, evicts cold ones.
K=2: page must be accessed 2+ times to be considered hot."""
from typing import Any, Dict, Optional
from collections import OrderedDict
import time


class BufferPool:
    """LRU-K(2) page cache."""

    def __init__(self, max_pages: int = 1024) -> None:
        self._max = max_pages
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._access_count: Dict[str, int] = {}
        self._hits = 0
        self._misses = 0

    def get(self, page_id: str) -> Optional[Any]:
        if page_id in self._cache:
            self._cache.move_to_end(page_id)
            self._access_count[page_id] = self._access_count.get(page_id, 0) + 1
            self._hits += 1
            return self._cache[page_id]
        self._misses += 1
        return None

    def put(self, page_id: str, data: Any) -> None:
        if page_id in self._cache:
            self._cache.move_to_end(page_id)
            self._cache[page_id] = data
            return
        if len(self._cache) >= self._max:
            self._evict()
        self._cache[page_id] = data
        self._access_count[page_id] = 1

    def _evict(self) -> None:
        # LRU-K: evict page with lowest access count, break ties by LRU
        if not self._cache:
            return
        min_count = float('inf')
        victim = None
        for pid in self._cache:
            cnt = self._access_count.get(pid, 0)
            if cnt < min_count:
                min_count = cnt; victim = pid
            if cnt <= 1:
                break  # Single-access pages are best candidates
        if victim:
            del self._cache[victim]
            self._access_count.pop(victim, None)
        else:
            self._cache.popitem(last=False)

    def invalidate(self, page_id: str) -> None:
        self._cache.pop(page_id, None)
        self._access_count.pop(page_id, None)

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total else 0.0

    @property
    def size(self) -> int:
        return len(self._cache)
