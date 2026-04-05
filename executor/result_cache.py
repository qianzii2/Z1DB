from __future__ import annotations
"""查询结果LRU缓存 — 表级版本失效。
同一SQL + 表未修改 → 直接返回缓存结果。
任何DML/DDL修改表 → 该表相关缓存全部失效。"""
from collections import OrderedDict
from typing import Any, Dict, Optional, Set
from metal.hash import murmur3_64


class ResultCache:
    """LRU结果缓存，语义失效。"""

    __slots__ = ('_cache', '_max_size', '_hits', '_misses')

    def __init__(self, max_size: int = 128) -> None:
        self._cache: OrderedDict[int, _CacheEntry] = OrderedDict()
        self._max_size = max_size
        self._hits = 0
        self._misses = 0

    @staticmethod
    def hash_sql(sql: str) -> int:
        """对SQL文本做哈希，供外部使用。"""
        return murmur3_64(sql.strip().lower().encode('utf-8'))

    def get(self, sql_hash: int, current_versions: Dict[str, int]) -> Optional[Any]:
        """查找缓存结果。current_versions = 调用方维护的表版本号。
        如果任何引用的表版本号变了，视为过期。"""
        if sql_hash not in self._cache:
            self._misses += 1
            return None
        entry = self._cache[sql_hash]
        # 检查所有引用表的版本号是否仍然匹配
        for table, ver in entry.table_versions.items():
            if current_versions.get(table, 0) != ver:
                del self._cache[sql_hash]
                self._misses += 1
                return None
        self._cache.move_to_end(sql_hash)
        self._hits += 1
        return entry.result

    def put(self, sql_hash: int, result: Any,
            table_versions: Dict[str, int]) -> None:
        """缓存查询结果。table_versions = {表名: 当前版本号}。"""
        if sql_hash in self._cache:
            self._cache.move_to_end(sql_hash)
            self._cache[sql_hash] = _CacheEntry(result, dict(table_versions))
            return
        if len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)
        self._cache[sql_hash] = _CacheEntry(result, dict(table_versions))

    def invalidate_table(self, table_name: str) -> None:
        """移除所有引用了该表的缓存条目。"""
        to_remove = []
        for key, entry in self._cache.items():
            if table_name in entry.table_versions:
                to_remove.append(key)
        for key in to_remove:
            del self._cache[key]

    def clear(self) -> None:
        self._cache.clear()

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def size(self) -> int:
        return len(self._cache)


class _CacheEntry:
    __slots__ = ('result', 'table_versions')

    def __init__(self, result: Any, table_versions: Dict[str, int]) -> None:
        self.result = result
        self.table_versions = table_versions
