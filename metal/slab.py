from __future__ import annotations
"""Slab allocator — skeleton for Phase 2+."""


class SlabAllocator:
    def __init__(self, object_size: int, slab_capacity: int = 4096) -> None:
        self.object_size = object_size
        self.slab_capacity = slab_capacity

    def alloc(self) -> int:
        return 0

    def free(self, slot_id: int) -> None:
        pass

    def write(self, slot_id: int, data: bytes) -> None:
        pass

    def read(self, slot_id: int) -> bytes:
        return b''
