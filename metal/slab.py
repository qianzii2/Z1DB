from __future__ import annotations
"""Slab allocator — O(1) alloc/free for fixed-size objects."""
import ctypes
from typing import Optional


class SlabAllocator:
    """Fixed-size object pool. alloc/free both O(1) via free-list stack."""

    __slots__ = ('_obj_size', '_slab_cap', '_buf', '_free_stack',
                 '_size', '_high_water')

    def __init__(self, object_size: int, slab_capacity: int = 4096) -> None:
        self._obj_size = object_size
        self._slab_cap = slab_capacity
        self._buf = ctypes.create_string_buffer(object_size * slab_capacity)
        # Free list: stack of available slot indices
        self._free_stack: list[int] = list(range(slab_capacity - 1, -1, -1))
        self._size = 0
        self._high_water = 0

    def alloc(self) -> int:
        """O(1) — returns slot_id."""
        if not self._free_stack:
            self._grow()
        slot = self._free_stack.pop()
        self._size += 1
        if slot >= self._high_water:
            self._high_water = slot + 1
        return slot

    def free(self, slot_id: int) -> None:
        """O(1) — returns slot to pool."""
        self._free_stack.append(slot_id)
        self._size -= 1

    def write(self, slot_id: int, data: bytes) -> None:
        offset = slot_id * self._obj_size
        n = min(len(data), self._obj_size)
        ctypes.memmove(ctypes.addressof(self._buf) + offset, data, n)

    def read(self, slot_id: int) -> bytes:
        offset = slot_id * self._obj_size
        return bytes(self._buf[offset: offset + self._obj_size])

    def read_into(self, slot_id: int, target: bytearray, target_offset: int = 0) -> None:
        """Zero-copy read into existing buffer."""
        offset = slot_id * self._obj_size
        target[target_offset: target_offset + self._obj_size] = \
            self._buf[offset: offset + self._obj_size]

    @property
    def size(self) -> int:
        return self._size

    @property
    def capacity(self) -> int:
        return self._slab_cap

    def _grow(self) -> None:
        old_cap = self._slab_cap
        new_cap = old_cap * 2
        new_buf = ctypes.create_string_buffer(self._obj_size * new_cap)
        ctypes.memmove(new_buf, self._buf, self._obj_size * old_cap)
        self._buf = new_buf
        self._free_stack = list(range(new_cap - 1, old_cap - 1, -1))
        self._slab_cap = new_cap
