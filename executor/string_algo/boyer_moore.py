from __future__ import annotations
"""Boyer-Moore string search — average O(n/m) for LIKE '%pattern%'.
Preprocesses pattern to skip large portions of text."""
from typing import List


class BoyerMoore:
    """Boyer-Moore with bad character + good suffix rules."""

    __slots__ = ('_pattern', '_m', '_bad_char', '_good_suffix')

    def __init__(self, pattern: str) -> None:
        self._pattern = pattern
        self._m = len(pattern)
        self._bad_char = self._build_bad_char()
        self._good_suffix = self._build_good_suffix()

    def search(self, text: str) -> List[int]:
        """Find all occurrences of pattern in text. Returns list of start positions."""
        n = len(text)
        m = self._m
        if m == 0:
            return list(range(n + 1))
        if m > n:
            return []

        positions: List[int] = []
        s = 0  # shift

        while s <= n - m:
            j = m - 1
            # Match from right to left
            while j >= 0 and self._pattern[j] == text[s + j]:
                j -= 1

            if j < 0:
                # Match found
                positions.append(s)
                s += self._good_suffix[0]
            else:
                # Shift by max of bad char and good suffix
                bc_shift = j - self._bad_char.get(text[s + j], -1)
                gs_shift = self._good_suffix[j + 1] if j + 1 < len(self._good_suffix) else 1
                s += max(bc_shift, gs_shift)

        return positions

    def contains(self, text: str) -> bool:
        """Check if pattern exists in text. Stops at first match."""
        n = len(text)
        m = self._m
        if m == 0:
            return True
        if m > n:
            return False

        s = 0
        while s <= n - m:
            j = m - 1
            while j >= 0 and self._pattern[j] == text[s + j]:
                j -= 1
            if j < 0:
                return True
            bc_shift = j - self._bad_char.get(text[s + j], -1)
            gs_shift = self._good_suffix[j + 1] if j + 1 < len(self._good_suffix) else 1
            s += max(bc_shift, gs_shift)
        return False

    def _build_bad_char(self) -> dict:
        """Last occurrence of each character in pattern."""
        table: dict = {}
        for i, ch in enumerate(self._pattern):
            table[ch] = i
        return table

    def _build_good_suffix(self) -> list:
        """Good suffix shift table."""
        m = self._m
        table = [0] * (m + 1)
        suffix = self._build_suffix()

        # Case 1: matching suffix exists elsewhere in pattern
        for i in range(m + 1):
            table[i] = m

        for i in range(m - 1, -1, -1):
            if suffix[i] == i + 1:
                for j in range(m - 1 - i):
                    if table[j] == m:
                        table[j] = m - 1 - i

        for i in range(m - 1):
            table[m - 1 - suffix[i]] = m - 1 - i

        return table

    def _build_suffix(self) -> list:
        """Compute suffix array for good suffix rule."""
        m = self._m
        suffix = [0] * m
        suffix[m - 1] = m
        g = m - 1
        f = 0
        for i in range(m - 2, -1, -1):
            if i > g and suffix[i + m - 1 - f] < i - g:
                suffix[i] = suffix[i + m - 1 - f]
            else:
                if i < g:
                    g = i
                f = i
                while g >= 0 and self._pattern[g] == self._pattern[g + m - 1 - f]:
                    g -= 1
                suffix[i] = f - g
        return suffix


def like_contains_search(text: str, pattern: str) -> bool:
    """Optimized LIKE '%pattern%' using Boyer-Moore."""
    bm = BoyerMoore(pattern)
    return bm.contains(text)
