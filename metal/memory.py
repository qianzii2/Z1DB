from __future__ import annotations
"""Raw memory block — skeleton for Phase 3."""


class RawMemoryBlock:
    """Placeholder. Will manage typed contiguous memory in Phase 3."""

    def __init__(self, dtype_code: str, capacity: int) -> None:
        self.dtype_code = dtype_code
        self.capacity = capacity
        self.size = 0
