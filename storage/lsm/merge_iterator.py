from __future__ import annotations
"""LSM 多路归并迭代器 — 合并 MemTable + 多层 SSTable。
O(n log k) 其中 k = 源数量。同 key 取最新源（source_id 最大）。
tombstone (value=None) 被跳过不输出。"""
import heapq
from typing import Any, Iterator, List, Optional, Tuple


class _Source:
    """一个有序数据源的迭代器包装。"""

    __slots__ = ('_iter', '_current', '_exhausted', '_id')

    def __init__(self, source_id: int,
                 iterator: Iterator[Tuple[Any, Any]]) -> None:
        self._id = source_id
        self._iter = iterator
        self._current: Optional[Tuple[Any, Any]] = None
        self._exhausted = False
        self._advance()

    def _advance(self) -> None:
        try:
            self._current = next(self._iter)
        except StopIteration:
            self._current = None
            self._exhausted = True

    @property
    def key(self):
        return self._current[0] if self._current else None

    @property
    def value(self):
        return self._current[1] if self._current else None

    @property
    def exhausted(self):
        return self._exhausted


class _HeapEntry:
    """堆条目。按 (key, -source_id) 排序。
    source_id 越大越新（后添加的 SSTable/MemTable 优先）。"""

    __slots__ = ('key', 'value', 'source_id')

    def __init__(self, key, value, source_id) -> None:
        self.key = key
        self.value = value
        self.source_id = source_id

    def __lt__(self, other: '_HeapEntry') -> bool:
        if self.key == other.key:
            # 同 key：source_id 大的排前面（更新的数据优先出堆）
            return self.source_id > other.source_id
        try:
            return self.key < other.key
        except TypeError:
            return str(self.key) < str(other.key)


class MergeIterator:
    """多路归并迭代器。
    合并多个有序 (key, value) 源，按 key 升序输出。
    同 key 取最新源的值。value=None 表示 tombstone（跳过）。"""

    def __init__(self, sources: List[Iterator[Tuple[Any, Any]]]
                 ) -> None:
        self._heap: List[_HeapEntry] = []
        self._sources: List[_Source] = []
        for sid, src in enumerate(sources):
            s = _Source(sid, src)
            self._sources.append(s)
            if not s.exhausted:
                heapq.heappush(
                    self._heap,
                    _HeapEntry(s.key, s.value, sid))

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[Any, Any]:
        """返回下一个 (key, value)。跳过 tombstone 和重复 key。"""
        while self._heap:
            entry = heapq.heappop(self._heap)
            key = entry.key
            best_value = entry.value
            best_sid = entry.source_id

            # 推进刚弹出的源
            src = self._sources[entry.source_id]
            src._advance()
            if not src.exhausted:
                heapq.heappush(
                    self._heap,
                    _HeapEntry(src.key, src.value,
                               entry.source_id))

            # 弹出所有同 key 条目，取 source_id 最大的
            while self._heap and self._heap[0].key == key:
                dup = heapq.heappop(self._heap)
                if dup.source_id > best_sid:
                    best_value = dup.value
                    best_sid = dup.source_id
                dup_src = self._sources[dup.source_id]
                dup_src._advance()
                if not dup_src.exhausted:
                    heapq.heappush(
                        self._heap,
                        _HeapEntry(dup_src.key,
                                   dup_src.value,
                                   dup.source_id))

            # 跳过 tombstone
            if best_value is not None:
                return (key, best_value)

        raise StopIteration
