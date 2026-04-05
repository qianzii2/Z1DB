from __future__ import annotations
"""Query result cache — returns cached results for unchanged tables."""
from typing import Any, Dict, Optional
from collections import OrderedDict
from metal.hash import murmur3_64


class ResultCache:
    """LRU cache for query results. Invalidated when tables are modified."""

    def __init__(self, max_entries: int = 128) -> None:
        self._cache: OrderedDict[int, tuple] = OrderedDict()
        # Each entry: (result, {table_name: version_at_cache_time})
        self._max_entries = max_entries
        self._hits = 0
        self._misses = 0

    def get(self, sql_hash: int, table_versions: Dict[str, int]) -> Optional[Any]:
        """Return cached result if all table versions match."""
        if sql_hash not in self._cache:
            self._misses += 1
            return None
        result, cached_versions = self._cache[sql_hash]
        # Check all referenced tables haven't changed
        for table, version in cached_versions.items():
            if table_versions.get(table, -1) != version:
                # Table was modified — invalidate this entry
                del self._cache[sql_hash]
                self._misses += 1
                return None
        self._cache.move_to_end(sql_hash)
        self._hits += 1
        return result

    def put(self, sql_hash: int, result: Any,
            table_versions: Dict[str, int]) -> None:
        """Cache a query result with table version snapshot."""
        if sql_hash in self._cache:
            self._cache.move_to_end(sql_hash)
        elif len(self._cache) >= self._max_entries:
            self._cache.popitem(last=False)
        self._cache[sql_hash] = (result, dict(table_versions))

    def invalidate_table(self, table_name: str) -> None:
        """Remove all cached entries that reference this table."""
        to_remove = []
        for sql_hash, (_, versions) in self._cache.items():
            if table_name in versions:
                to_remove.append(sql_hash)
        for h in to_remove:
            del self._cache[h]

    def clear(self) -> None:
        self._cache.clear()

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def size(self) -> int:
        return len(self._cache)

    @staticmethod
    def hash_sql(sql: str) -> int:
        return murmur3_64(sql.strip().encode('utf-8'))
