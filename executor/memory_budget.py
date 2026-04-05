from __future__ import annotations
"""内存预算 — 每算子配额，超限触发溢写。
维护running total避免每次sum()遍历。"""
import threading
from typing import Dict, Optional
from metal.config import DEFAULT_MEMORY_LIMIT


class MemoryBudget:
    """跟踪所有活跃算子的内存分配。"""

    __slots__ = ('_total_limit', '_allocations', '_mutex', '_peak',
                 '_current_total')

    def __init__(self, total_limit: int = DEFAULT_MEMORY_LIMIT) -> None:
        self._total_limit = total_limit
        self._allocations: Dict[str, int] = {}
        self._mutex = threading.Lock()
        self._peak = 0
        self._current_total = 0  # 维护running total

    @property
    def total_used(self) -> int:
        with self._mutex:
            return self._current_total

    @property
    def total_limit(self) -> int:
        return self._total_limit

    @property
    def peak_usage(self) -> int:
        return self._peak

    def request(self, operator_name: str, bytes_needed: int) -> bool:
        """请求内存。在预算内返回True，应溢写返回False。"""
        with self._mutex:
            if self._current_total + bytes_needed > self._total_limit:
                return False
            old = self._allocations.get(operator_name, 0)
            self._allocations[operator_name] = old + bytes_needed
            self._current_total += bytes_needed
            if self._current_total > self._peak:
                self._peak = self._current_total
            return True

    def release(self, operator_name: str) -> None:
        """释放算子的所有内存。"""
        with self._mutex:
            released = self._allocations.pop(operator_name, 0)
            self._current_total -= released

    def release_partial(self, operator_name: str,
                        bytes_released: int) -> None:
        """部分释放内存。"""
        with self._mutex:
            old = self._allocations.get(operator_name, 0)
            actual = min(bytes_released, old)
            self._allocations[operator_name] = old - actual
            self._current_total -= actual
            if self._allocations[operator_name] <= 0:
                self._allocations.pop(operator_name, None)

    def should_spill(self, operator_name: str,
                     estimated_bytes: int) -> bool:
        """检查是否应溢写（不实际分配）。"""
        with self._mutex:
            return (self._current_total + estimated_bytes
                    > self._total_limit)

    def estimate_row_bytes(self, row_count: int,
                           avg_row_width: int = 100) -> int:
        return row_count * avg_row_width

    def report(self) -> Dict[str, int]:
        with self._mutex:
            return dict(self._allocations)

    def reset(self) -> None:
        """重置所有计数。"""
        with self._mutex:
            self._allocations.clear()
            self._current_total = 0
