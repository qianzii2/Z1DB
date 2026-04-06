from __future__ import annotations
"""查询历史记录。"""
import time
from typing import Any, Dict, List


class QueryHistory:
    def __init__(self, max_size: int = 50) -> None:
        self._entries: List[Dict[str, Any]] = []
        self._max = max_size

    def add(self, sql: str, timing: float = 0.0, rows: int = 0) -> None:
        self._entries.append({
            'sql': sql.strip()[:200], 'timing': timing,
            'rows': rows, 'ts': time.time()})
        if len(self._entries) > self._max:
            self._entries.pop(0)

    @property
    def entries(self) -> List[Dict[str, Any]]:
        return self._entries

    def clear(self) -> None:
        self._entries.clear()
