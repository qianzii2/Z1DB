from __future__ import annotations
"""败者树 — K路归并O(n log K)。修正_replay为真正的O(log K)沿树上行。"""
from typing import Any, Callable, Iterator, List, Optional, Tuple

_SENTINEL = object()


class LoserTree:
    """K路归并败者树。每次pop_winner()代价O(log K)。"""

    __slots__ = ('_k', '_tree', '_leaves', '_sources', '_key_fn',
                 '_exhausted', '_winner')

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

    def _init(self) -> None:
        # 从每个源读取首元素
        for i in range(self._k):
            val = next(self._sources[i], _SENTINEL)
            self._leaves[i] = val
            if val is _SENTINEL:
                self._exhausted[i] = True

        if self._k <= 1:
            self._winner = 0
            return

        # 初始化内部节点（全设为0）
        self._tree = [0] * self._k

        # 逐个"比赛"建树
        winner = 0
        for i in range(1, self._k):
            if self._is_less(i, winner):
                self._tree[i] = winner  # loser存入树节点
                winner = i
            else:
                self._tree[i] = i
        # 最终winner不存入树，单独记录
        self._winner = winner

    def _is_less(self, i: int, j: int) -> bool:
        """i是否应排在j前面。"""
        if self._exhausted[i]:
            return False
        if self._exhausted[j]:
            return True
        try:
            return self._key_fn(self._leaves[i]) <= self._key_fn(
                self._leaves[j])
        except TypeError:
            return str(self._leaves[i]) <= str(self._leaves[j])

    def pop_winner(self) -> Optional[Tuple[Any, int]]:
        """弹出当前最小值。返回(value, source_index)或None。"""
        wi = self._winner
        if self._exhausted[wi]:
            return None
        value = self._leaves[wi]
        source_idx = wi

        # 推进winner的源
        next_val = next(self._sources[wi], _SENTINEL)
        if next_val is _SENTINEL:
            self._exhausted[wi] = True
        self._leaves[wi] = next_val

        # 沿树上行重赛 — 真正的O(log K)
        self._replay(wi)
        return (value, source_idx)

    def _replay(self, idx: int) -> None:
        """从叶子idx向上重赛。O(log K)。"""
        if self._k <= 1:
            self._winner = 0
            return

        winner = idx
        # 从叶子位置向根遍历
        pos = (idx + self._k) // 2  # 父节点位置
        while pos > 0 and pos < self._k:
            loser_idx = self._tree[pos]
            if self._is_less(loser_idx, winner):
                # loser赢了当前winner → 交换
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
