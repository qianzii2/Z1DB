from __future__ import annotations
"""LRU 编译缓存 — 复用已编译的表达式函数。
[P11] hash_expr 限制 repr 长度，避免大 AST 哈希过慢。"""
from typing import Any, Callable, Dict, Optional
from collections import OrderedDict
from metal.hash import z1hash64


class CompileCache:
    """编译缓存。key = AST 内容哈希。"""

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
            self._cache.popitem(last=False)
        self._cache[expr_hash] = fn

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @staticmethod
    def hash_expr(expr: Any) -> int:
        """哈希 AST 表达式。
        [P11] 限制 repr 长度为 500 字符，避免大 AST 哈希过慢。"""
        r = repr(expr)
        if len(r) > 500:
            r = r[:500]
        return z1hash64(r.encode('utf-8'))
