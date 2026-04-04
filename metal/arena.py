from __future__ import annotations
"""Arena allocator — skeleton for Phase 3."""

from metal.config import ARENA_BLOCK_SIZE


class Arena:
    """Placeholder bump allocator. Phase 3 will use ctypes buffers."""

    def __init__(self) -> None:
        self._used = 0

    def alloc(self, size: int) -> tuple[None, int]:  # type: ignore[return]
        offset = self._used
        self._used += size
        return (None, offset)

    def reset(self) -> None:
        self._used = 0

    def bytes_used(self) -> int:
        return self._used
