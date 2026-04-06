from __future__ import annotations
"""败者树 — K 路归并 O(n log K)。
每次 pop_winner 代价 O(log K)，沿树上行重赛。
用于 ExternalSort 和 LSM Compaction 的多路归并。"""
from typing import Any, Callable, Iterator, List, Optional, Tuple

_SENTINEL = object()


class LoserTree:
    """K 路归并败者树。内部节点存"败者"，winner 单独记录。"""

    __slots__ = ('_k', '_tree', '_leaves', '_sources',
                 '_key_fn', '_exhausted', '_winner')

    def __init__(self, sources: List[Iterator],
                 key_fn: Callable = lambda x: x) -> None:
        self._k = len(sources)
        self._sources = sources
        self._key_fn = key_fn
        self._tree: List[int] = [0] * self._k
        self._leaves: List[Any] = [_SENTINEL] * self._k
        self._exhausted = [False] * self._k
        self._winner = 0
        self._init()

    def _init(self):
        # 从每个源读取首元素
        for i in range(self._k):
            val = next(self._sources[i], _SENTINEL)
            self._leaves[i] = val
            if val is _SENTINEL:
                self._exhausted[i] = True
        if self._k <= 1:
            self._winner = 0
            return
        self._tree = [0] * self._k
        # 逐个比赛建树
        winner = 0
        for i in range(1, self._k):
            if self._is_less(i, winner):
                self._tree[i] = winner
                winner = i
            else:
                self._tree[i] = i
        self._winner = winner

    def _is_less(self, i: int, j: int) -> bool:
        """i 是否应排在 j 前面。已耗尽的源排最后。"""
        if self._exhausted[i]:
            return False
        if self._exhausted[j]:
            return True
        try:
            return (self._key_fn(self._leaves[i])
                    <= self._key_fn(self._leaves[j]))
        except TypeError:
            return (str(self._leaves[i])
                    <= str(self._leaves[j]))

    def pop_winner(self) -> Optional[Tuple[Any, int]]:
        """弹出当前最小值。返回 (value, source_index) 或 None。"""
        wi = self._winner
        if self._exhausted[wi]:
            return None
        value = self._leaves[wi]
        source_idx = wi
        # 推进 winner 的源
        next_val = next(self._sources[wi], _SENTINEL)
        if next_val is _SENTINEL:
            self._exhausted[wi] = True
        self._leaves[wi] = next_val
        # 沿树上行重赛 — O(log K)
        self._replay(wi)
        return (value, source_idx)

    def _replay(self, idx: int):
        """从叶子 idx 向根重赛。O(log K)。"""
        if self._k <= 1:
            self._winner = 0
            return
        winner = idx
        pos = (idx + self._k) // 2
        while pos > 0 and pos < self._k:
            loser_idx = self._tree[pos]
            if self._is_less(loser_idx, winner):
                self._tree[pos] = winner
                winner = loser_idx
            pos //= 2
        self._winner = winner

    def is_exhausted(self) -> bool:
        return all(self._exhausted)

    def merge_all(self) -> List[Any]:
        """排空所有源到有序列表。"""
        result = []
        while True:
            item = self.pop_winner()
            if item is None:
                break
            result.append(item[0])
        return result
