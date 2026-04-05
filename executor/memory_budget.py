from __future__ import annotations
"""Memory budget — tracks per-operator memory usage, triggers spill."""
from typing import Dict
from metal.config import DEFAULT_MEMORY_LIMIT


class MemoryBudget:
    """Tracks memory usage per operator. Suggests spill when over budget."""

    def __init__(self, total_limit: int = DEFAULT_MEMORY_LIMIT) -> None:
        self._total_limit = total_limit
        self._allocated: Dict[str, int] = {}

    def allocate(self, operator_name: str, bytes_needed: int) -> bool:
        """Try to allocate memory. Returns False if would exceed budget."""
        current_total = sum(self._allocated.values())
        if current_total + bytes_needed > self._total_limit:
            return False
        self._allocated[operator_name] = self._allocated.get(operator_name, 0) + bytes_needed
        return True

    def release(self, operator_name: str) -> None:
        self._allocated.pop(operator_name, None)

    def is_over_budget(self) -> bool:
        return sum(self._allocated.values()) > self._total_limit

    def should_spill(self, operator_name: str, estimated_bytes: int) -> bool:
        """Should this operator switch to disk-spilling mode?"""
        current = sum(self._allocated.values())
        return current + estimated_bytes > self._total_limit * 0.8  # 80% threshold

    def estimate_rows_in_budget(self, row_width: int) -> int:
        """How many rows can fit in remaining budget?"""
        remaining = self._total_limit - sum(self._allocated.values())
        return max(1, remaining // max(row_width, 1))

    @property
    def used(self) -> int:
        return sum(self._allocated.values())

    @property
    def remaining(self) -> int:
        return max(0, self._total_limit - self.used)

    @property
    def utilization(self) -> float:
        return self.used / self._total_limit if self._total_limit > 0 else 0.0
