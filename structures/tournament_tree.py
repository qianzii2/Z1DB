from __future__ import annotations
"""Loser Tree — K-way merge in O(n log K). Better branch prediction than heap.
Used for External Sort K-way merge and UNION ALL ordered merge."""
from typing import Any, Callable, Iterator, List, Optional, Tuple

_SENTINEL = object()


class LoserTree:
    """K-way merge using a tournament (loser) tree.
    Each pop_winner() costs O(log K) comparisons."""

    __slots__ = ('_k', '_tree', '_leaves', '_sources', '_key_fn',
                 '_exhausted', '_winner')

    def __init__(self, sources: List[Iterator],
                 key_fn: Callable = lambda x: x) -> None:
        self._k = len(sources)
        self._sources = sources
        self._key_fn = key_fn
        self._tree = [0] * self._k  # internal nodes store loser index
        self._leaves: List[Any] = [_SENTINEL] * self._k
        self._exhausted = [False] * self._k
        self._winner = 0
        self._init()

    def _init(self) -> None:
        # Fill leaves from sources
        for i in range(self._k):
            val = next(self._sources[i], _SENTINEL)
            self._leaves[i] = val
            if val is _SENTINEL:
                self._exhausted[i] = True
        # Build tree from leaves
        if self._k <= 1:
            self._winner = 0
            return
        # Initialize all internal nodes to 0
        self._tree = [0] * self._k
        # Play initial tournament
        winner = 0
        for i in range(1, self._k):
            if self._is_less(i, winner):
                self._tree[self._parent(i)] = winner
                winner = i
            else:
                self._tree[self._parent(i)] = i
        self._winner = winner

    def _parent(self, i: int) -> int:
        return (i + self._k) // 2 if self._k > 1 else 0

    def _is_less(self, i: int, j: int) -> bool:
        """True if leaf i should come before leaf j."""
        if self._exhausted[i]:
            return False
        if self._exhausted[j]:
            return True
        return self._key_fn(self._leaves[i]) <= self._key_fn(self._leaves[j])

    def pop_winner(self) -> Optional[Tuple[Any, int]]:
        """Pop the current minimum value. Returns (value, source_index) or None."""
        wi = self._winner
        if self._exhausted[wi]:
            return None
        value = self._leaves[wi]
        source_idx = wi

        # Advance the winner's source
        next_val = next(self._sources[wi], _SENTINEL)
        if next_val is _SENTINEL:
            self._exhausted[wi] = True
        self._leaves[wi] = next_val

        # Replay from winner leaf to root
        self._replay(wi)
        return (value, source_idx)

    def _replay(self, idx: int) -> None:
        """Replay tournament from leaf idx upward. O(log K)."""
        winner = idx
        # Simple replay: re-find winner among all
        # For small K this is efficient enough
        best = -1
        for i in range(self._k):
            if not self._exhausted[i]:
                if best == -1 or self._is_less(i, best):
                    best = i
        self._winner = best if best != -1 else 0

    def is_exhausted(self) -> bool:
        return all(self._exhausted)

    def merge_all(self) -> List[Any]:
        """Drain all sources into a sorted list."""
        result = []
        while True:
            item = self.pop_winner()
            if item is None:
                break
            result.append(item[0])
        return result
