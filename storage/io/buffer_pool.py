from __future__ import annotations

"""LRU-K(2) 分页缓冲池。
[P13] SSTable 按偏移分页缓存，每页 16KB。
热页面驻留内存，冷页面 LRU-K 淘汰。"""
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple
import os

# 每页大小（字节）
PAGE_SIZE = 16 * 1024
# 默认最大缓存页数
DEFAULT_MAX_PAGES = 1024


class PageId:
    """页面标识：(文件路径, 页内偏移)。"""
    __slots__ = ('file_path', 'page_offset')

    def __init__(self, file_path: str, page_offset: int) -> None:
        self.file_path = file_path
        self.page_offset = page_offset

    def __hash__(self) -> int:
        return hash((self.file_path, self.page_offset))

    def __eq__(self, other) -> bool:
        if not isinstance(other, PageId): return False
        return (self.file_path == other.file_path
                and self.page_offset == other.page_offset)

    def __repr__(self) -> str:
        return f"Page({self.file_path}:{self.page_offset})"


class CachedPage:
    """缓存页：数据 + 访问计数。"""
    __slots__ = ('data', 'access_count', 'dirty')

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.access_count = 1
        self.dirty = False


class BufferPool:
    """LRU-K(2) 分页缓冲池。

    接口:
      get(page_id) → data or None        # 查缓存
      put(page_id, data)                   # 写入缓存
      read_page(file_path, offset) → data  # 读文件页（自动缓存）
      invalidate(file_path)                # 失效文件的所有页
    """

    def __init__(self, max_pages: int = DEFAULT_MAX_PAGES) -> None:
        self._max = max_pages
        self._cache: OrderedDict[PageId, CachedPage] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, page_id: Any) -> Optional[Any]:
        """查缓存。支持 PageId 或字符串 key。"""
        if isinstance(page_id, str):
            # 兼容旧接口：字符串 key → 整体缓存
            key = PageId(page_id, 0)
        else:
            key = page_id
        if key in self._cache:
            entry = self._cache[key]
            entry.access_count += 1
            self._cache.move_to_end(key)
            self._hits += 1
            return entry.data
        self._misses += 1
        return None

    def put(self, page_id: Any, data: Any) -> None:
        """写入缓存。"""
        if isinstance(page_id, str):
            key = PageId(page_id, 0)
        else:
            key = page_id
        if key in self._cache:
            self._cache[key].data = data
            self._cache.move_to_end(key)
            return
        if len(self._cache) >= self._max:
            self._evict()
        self._cache[key] = CachedPage(data)

    def read_page(self, file_path: str, offset: int) -> Optional[bytes]:
        """[P13] 读取文件中指定偏移的页。自动缓存。"""
        pid = PageId(file_path, offset)
        cached = self.get(pid)
        if cached is not None:
            return cached
        # 缓存未命中：从文件读取
        try:
            with open(file_path, 'rb') as f:
                f.seek(offset)
                data = f.read(PAGE_SIZE)
            if data:
                self.put(pid, data)
            return data
        except (FileNotFoundError, OSError):
            return None

    def read_range(self, file_path: str, offset: int,
                   length: int) -> Optional[bytes]:
        """[P13] 读取跨页范围。自动分页缓存。"""
        result = bytearray()
        current_offset = offset
        remaining = length

        while remaining > 0:
            # 对齐到页边界
            page_start = (current_offset // PAGE_SIZE) * PAGE_SIZE
            page_offset_within = current_offset - page_start
            page_data = self.read_page(file_path, page_start)
            if page_data is None:
                break
            # 从页内提取需要的字节
            available = len(page_data) - page_offset_within
            to_read = min(remaining, available)
            if to_read <= 0:
                break
            result.extend(page_data[page_offset_within:page_offset_within + to_read])
            current_offset += to_read
            remaining -= to_read

        return bytes(result) if result else None

    def invalidate(self, file_or_page: Any) -> None:
        """失效文件或页面的缓存。"""
        if isinstance(file_or_page, str):
            # 失效文件的所有页
            to_remove = [k for k in self._cache
                         if k.file_path == file_or_page]
            # 也检查旧接口的字符串 key
            old_key = PageId(file_or_page, 0)
            if old_key in self._cache:
                to_remove.append(old_key)
            for k in to_remove:
                del self._cache[k]
        elif isinstance(file_or_page, PageId):
            self._cache.pop(file_or_page, None)

    def _evict(self) -> None:
        """LRU-K(2) 淘汰：优先淘汰访问次数最少的页。"""
        if not self._cache:
            return
        # 找访问次数最少的页
        min_count = float('inf')
        victim = None
        for pid, page in self._cache.items():
            if page.access_count < min_count:
                min_count = page.access_count
                victim = pid
            if page.access_count <= 1:
                break  # 单次访问的页最适合淘汰
        if victim:
            del self._cache[victim]
        else:
            self._cache.popitem(last=False)

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total else 0.0

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def memory_bytes(self) -> int:
        """估算缓存占用的内存字节数。"""
        total = 0
        for page in self._cache.values():
            if isinstance(page.data, (bytes, bytearray)):
                total += len(page.data)
            else:
                total += 1024  # 估算非字节类型
        return total
