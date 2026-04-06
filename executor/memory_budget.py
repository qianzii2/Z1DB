from __future__ import annotations
"""内存预算 — 每算子配额 + Slab 分配跟踪。
集成 slab.py：固定大小算子状态用 Slab 分配。"""
import threading
from typing import Any, Dict, Optional
from metal.config import DEFAULT_MEMORY_LIMIT

try:
    from metal.slab import SlabAllocator
    _HAS_SLAB = True
except ImportError:
    _HAS_SLAB = False

# 算子状态块大小（字节）
_OPERATOR_STATE_SIZE = 256


class MemoryBudget:
    """跟踪所有活跃算子的内存分配。
    集成 Slab：固定大小算子状态 O(1) 分配/释放。"""

    __slots__ = ('_total_limit', '_allocations', '_mutex',
                 '_peak', '_current_total', '_slab',
                 '_slab_slots')

    def __init__(self,
                 total_limit: int = DEFAULT_MEMORY_LIMIT
                 ) -> None:
        self._total_limit = total_limit
        self._allocations: Dict[str, int] = {}
        self._mutex = threading.Lock()
        self._peak = 0
        self._current_total = 0
        # Slab 分配器：管理固定大小的算子状态
        self._slab: Optional[SlabAllocator] = None
        self._slab_slots: Dict[str, int] = {}  # name → slot_id
        if _HAS_SLAB:
            try:
                self._slab = SlabAllocator(
                    _OPERATOR_STATE_SIZE, slab_capacity=256)
            except Exception:
                self._slab = None

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

    def request(self, operator_name: str,
                bytes_needed: int) -> bool:
        """请求内存。超预算返回 False。"""
        with self._mutex:
            if (self._current_total + bytes_needed
                    > self._total_limit):
                return False
            old = self._allocations.get(operator_name, 0)
            self._allocations[operator_name] = (
                old + bytes_needed)
            self._current_total += bytes_needed
            if self._current_total > self._peak:
                self._peak = self._current_total
            return True

    def alloc_operator_state(self,
                             operator_name: str) -> int:
        """用 Slab 分配固定大小算子状态。返回 slot_id。"""
        if self._slab is None:
            return -1
        with self._mutex:
            try:
                slot = self._slab.alloc()
                self._slab_slots[operator_name] = slot
                self._current_total += _OPERATOR_STATE_SIZE
                if self._current_total > self._peak:
                    self._peak = self._current_total
                return slot
            except Exception:
                return -1

    def free_operator_state(self,
                            operator_name: str) -> None:
        """释放 Slab 分配的算子状态。"""
        if self._slab is None:
            return
        with self._mutex:
            slot = self._slab_slots.pop(operator_name, -1)
            if slot >= 0:
                self._slab.free(slot)
                self._current_total -= _OPERATOR_STATE_SIZE

    def release(self, operator_name: str) -> None:
        """释放算子的所有内存。"""
        with self._mutex:
            released = self._allocations.pop(
                operator_name, 0)
            self._current_total -= released
            # 同时释放 Slab 槽位
            slot = self._slab_slots.pop(operator_name, -1)
            if slot >= 0 and self._slab:
                self._slab.free(slot)
                self._current_total -= _OPERATOR_STATE_SIZE

    def release_partial(self, operator_name: str,
                        bytes_released: int) -> None:
        with self._mutex:
            old = self._allocations.get(operator_name, 0)
            actual = min(bytes_released, old)
            self._allocations[operator_name] = old - actual
            self._current_total -= actual
            if self._allocations[operator_name] <= 0:
                self._allocations.pop(operator_name, None)

    def should_spill(self, operator_name: str,
                     estimated_bytes: int) -> bool:
        with self._mutex:
            return (self._current_total + estimated_bytes
                    > self._total_limit)

    def estimate_row_bytes(self, row_count: int,
                           avg_row_width: int = 100) -> int:
        return row_count * avg_row_width

    def report(self) -> Dict[str, Any]:
        with self._mutex:
            result = dict(self._allocations)
            if self._slab:
                result['__slab_used'] = self._slab.size
                result['__slab_capacity'] = self._slab.capacity
            return result

    def reset(self) -> None:
        with self._mutex:
            self._allocations.clear()
            self._current_total = 0
            self._slab_slots.clear()
            if _HAS_SLAB:
                try:
                    self._slab = SlabAllocator(
                        _OPERATOR_STATE_SIZE,
                        slab_capacity=256)
                except Exception:
                    self._slab = None
