from __future__ import annotations
"""查询结果 LRU 缓存。
[P25] SQL 标准化：字符串外空白压缩，字符串内保留原样。"""
import re
from collections import OrderedDict
from typing import Any, Dict, Optional
from metal.hash import z1hash64

_WHITESPACE_RE = re.compile(r'\s+')


class ResultCache:
    __slots__ = ('_cache', '_max_size', '_hits', '_misses')

    def __init__(self, max_size: int = 128) -> None:
        self._cache: OrderedDict[int, _CacheEntry] = OrderedDict()
        self._max_size = max_size
        self._hits = 0
        self._misses = 0

    @staticmethod
    def hash_sql(sql: str) -> int:
        """[P25] 标准化 SQL 后哈希。
        - 字符串外：多余空白压缩为单个空格
        - 单引号和双引号内：保持原样
        """
        parts: list = []
        in_quote = False
        quote_char = ''
        current: list = []
        stripped = sql.strip()
        i = 0
        while i < len(stripped):
            ch = stripped[i]
            if not in_quote:
                if ch in ("'", '"'):
                    # 先标准化非字符串部分
                    text = ''.join(current)
                    parts.append(_WHITESPACE_RE.sub(' ', text))
                    current = [ch]
                    in_quote = True
                    quote_char = ch
                else:
                    current.append(ch)
            else:
                current.append(ch)
                if ch == quote_char:
                    # 检查转义引号（'' 或 ""）
                    if i + 1 < len(stripped) and stripped[i + 1] == quote_char:
                        current.append(quote_char)
                        i += 1
                    else:
                        # 引号结束——原样保留
                        parts.append(''.join(current))
                        current = []
                        in_quote = False
                        quote_char = ''
            i += 1
        # 处理剩余
        if current:
            if in_quote:
                parts.append(''.join(current))
            else:
                parts.append(_WHITESPACE_RE.sub(' ', ''.join(current)))
        normalized = ''.join(parts)
        return z1hash64(normalized.encode('utf-8'))

    def get(self, sql_hash: int,
            current_versions: Dict[str, int]) -> Optional[Any]:
        if sql_hash not in self._cache:
            self._misses += 1
            return None
        entry = self._cache[sql_hash]
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
        if sql_hash in self._cache:
            self._cache.move_to_end(sql_hash)
            self._cache[sql_hash] = _CacheEntry(
                result, dict(table_versions))
            return
        if len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)
        self._cache[sql_hash] = _CacheEntry(
            result, dict(table_versions))

    def invalidate_table(self, table_name: str) -> None:
        to_remove = [k for k, e in self._cache.items()
                     if table_name in e.table_versions]
        for k in to_remove:
            del self._cache[k]

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

    def __init__(self, result: Any,
                 table_versions: Dict[str, int]) -> None:
        self.result = result
        self.table_versions = table_versions
