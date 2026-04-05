from __future__ import annotations
"""Aho-Corasick — multi-pattern matching O(n + m + z).
Paper: Aho & Corasick, 1975.
One scan finds ALL patterns. Used for IN ('a','b','c') on text columns."""
from typing import Dict, List, Optional, Set, Tuple
from collections import deque


class AhoCorasick:
    """Multi-pattern automaton. Build once, search many texts."""

    def __init__(self, patterns: List[str]) -> None:
        self._goto: List[Dict[str, int]] = [{}]  # state → {char → next_state}
        self._fail: List[int] = [0]               # failure links
        self._output: List[Set[int]] = [set()]    # state → set of pattern indices
        self._patterns = patterns
        self._build()

    def _build(self) -> None:
        # Phase 1: Build goto function (trie)
        for pi, pattern in enumerate(self._patterns):
            state = 0
            for ch in pattern:
                if ch not in self._goto[state]:
                    new_state = len(self._goto)
                    self._goto.append({})
                    self._fail.append(0)
                    self._output.append(set())
                    self._goto[state][ch] = new_state
                state = self._goto[state][ch]
            self._output[state].add(pi)

        # Phase 2: Build failure links (BFS)
        queue: deque = deque()
        # Initialize depth-1 states
        for ch, s in self._goto[0].items():
            self._fail[s] = 0
            queue.append(s)

        while queue:
            r = queue.popleft()
            for ch, s in self._goto[r].items():
                queue.append(s)
                state = self._fail[r]
                while state != 0 and ch not in self._goto[state]:
                    state = self._fail[state]
                self._fail[s] = self._goto[state].get(ch, 0)
                if self._fail[s] == s:
                    self._fail[s] = 0  # prevent self-loop
                self._output[s] = self._output[s] | self._output[self._fail[s]]

    def search(self, text: str) -> List[Tuple[int, str]]:
        """Find all pattern occurrences. Returns [(position, pattern_str), ...]."""
        results: List[Tuple[int, str]] = []
        state = 0
        for i, ch in enumerate(text):
            while state != 0 and ch not in self._goto[state]:
                state = self._fail[state]
            state = self._goto[state].get(ch, 0)
            for pi in self._output[state]:
                pattern = self._patterns[pi]
                results.append((i - len(pattern) + 1, pattern))
        return results

    def contains_any(self, text: str) -> bool:
        """Check if text contains ANY of the patterns. Stops at first match."""
        state = 0
        for ch in text:
            while state != 0 and ch not in self._goto[state]:
                state = self._fail[state]
            state = self._goto[state].get(ch, 0)
            if self._output[state]:
                return True
        return False

    def which_patterns(self, text: str) -> Set[int]:
        """Return set of pattern indices found in text."""
        found: Set[int] = set()
        state = 0
        for ch in text:
            while state != 0 and ch not in self._goto[state]:
                state = self._fail[state]
            state = self._goto[state].get(ch, 0)
            found |= self._output[state]
        return found
