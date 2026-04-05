from __future__ import annotations

"""Arena allocator — O(1) alloc, O(1) bulk free, zero fragmentation."""
import ctypes


class Arena:
    """Bump allocator. All allocations freed at once via reset().

    Key invariant: self._blocks holds strong references to prevent GC.
    """

    BLOCK_SIZE = 1024 * 1024  # 1 MB

    __slots__ = ('_blocks', '_current', '_offset', '_total_used')

    def __init__(self) -> None:
        self._blocks: list = []
        self._current = ctypes.create_string_buffer(self.BLOCK_SIZE)
        self._blocks.append(self._current)
        self._offset = 0
        self._total_used = 0

    def alloc(self, size: int) -> tuple:
        """O(1) allocation. Returns (buffer, offset_within_buffer)."""
        if size <= 0:
            return (self._current, self._offset)

        # Align to 8 bytes
        aligned = (size + 7) & ~7

        if self._offset + aligned > len(self._current):
            # Current block exhausted — allocate new one
            block_size = max(aligned, self.BLOCK_SIZE)
            self._current = ctypes.create_string_buffer(block_size)
            self._blocks.append(self._current)
            self._offset = 0

        offset = self._offset
        self._offset += aligned
        self._total_used += aligned
        return (self._current, offset)

    def alloc_and_write(self, data: bytes) -> tuple:
        """Alloc + copy data in one call."""
        buf, offset = self.alloc(len(data))
        ctypes.memmove(ctypes.addressof(buf) + offset, data, len(data))
        return (buf, offset)

    def reset(self) -> None:
        """Free all allocations. O(1). Caller must ensure no external refs."""
        self._blocks.clear()
        self._current = ctypes.create_string_buffer(self.BLOCK_SIZE)
        self._blocks.append(self._current)
        self._offset = 0
        self._total_used = 0

    def bytes_used(self) -> int:
        return self._total_used

    def block_count(self) -> int:
        return len(self._blocks)
