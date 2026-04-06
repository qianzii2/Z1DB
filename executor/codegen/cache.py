from __future__ import annotations
"""LRU compile cache — reuse compiled functions for repeated query patterns."""
from typing import Any, Callable, Dict, Optional
from collections import OrderedDict
from metal.hash import murmur3_64


class CompileCache:
    """LRU cache for compiled expressions. Key = AST content hash."""

    def __init__(self, max_size: int = 256) -> None:
        self._cache: OrderedDict[int, Callable] = OrderedDict()
        self._max_size = max_size
        self._hits = 0
        self._misses = 0

    def get(self, expr_hash: int) -> Optional[Callable]:
        if expr_hash in self._cache:
            self._cache.move_to_end(expr_hash)
            self._hits += 1
            return self._cache[expr_hash]
        self._misses += 1
        return None

    def put(self, expr_hash: int, fn: Callable) -> None:
        if expr_hash in self._cache:
            self._cache.move_to_end(expr_hash)
            self._cache[expr_hash] = fn
            return
        if len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)  # evict LRU
        self._cache[expr_hash] = fn

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @staticmethod
    def hash_expr(expr: Any) -> int:
        """Hash an AST expression for cache lookup."""
        return murmur3_64(repr(expr).encode('utf-8'))
