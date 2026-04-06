from __future__ import annotations
"""Arena 分配器 — O(1) 分配、O(1) 批量释放、零碎片。
适用场景：算子的 open→close 周期内分配大量小对象。
用法:
    with Arena() as arena:
        buf, off = arena.alloc(1024)
        # ... 使用内存 ...
    # 退出 with 后自动释放全部内存
"""
import ctypes


class Arena:
    """Bump 分配器。所有分配在 reset() 时一次性释放。
    _blocks 持有强引用防止 GC 回收。"""

    BLOCK_SIZE = 1024 * 1024  # 1 MB

    __slots__ = ('_blocks', '_current', '_offset', '_total_used')

    def __init__(self) -> None:
        self._blocks: list = []
        self._current = ctypes.create_string_buffer(self.BLOCK_SIZE)
        self._blocks.append(self._current)
        self._offset = 0
        self._total_used = 0

    # ═══ 上下文管理器 [D03] ═══

    def __enter__(self) -> 'Arena':
        return self

    def __exit__(self, *args) -> None:
        self.reset()

    # ═══ 分配 ═══

    def alloc(self, size: int) -> tuple:
        """O(1) 分配。返回 (buffer, offset_within_buffer)。
        8 字节对齐，确保数值类型的内存对齐。"""
        if size <= 0:
            return (self._current, self._offset)

        # 对齐到 8 字节
        aligned = (size + 7) & ~7

        if self._offset + aligned > len(self._current):
            # 当前块不够 → 分配新块
            block_size = max(aligned, self.BLOCK_SIZE)
            self._current = ctypes.create_string_buffer(block_size)
            self._blocks.append(self._current)
            self._offset = 0

        offset = self._offset
        self._offset += aligned
        self._total_used += aligned
        return (self._current, offset)

    def alloc_and_write(self, data: bytes) -> tuple:
        """分配并写入数据。返回 (buffer, offset)。"""
        buf, offset = self.alloc(len(data))
        ctypes.memmove(ctypes.addressof(buf) + offset, data, len(data))
        return (buf, offset)

    # ═══ 释放 ═══

    def reset(self) -> None:
        """释放全部内存。O(1)。调用方需确保无外部引用残留。"""
        self._blocks.clear()
        self._current = ctypes.create_string_buffer(self.BLOCK_SIZE)
        self._blocks.append(self._current)
        self._offset = 0
        self._total_used = 0

    # ═══ 统计 ═══

    def bytes_used(self) -> int:
        """已分配的总字节数。"""
        return self._total_used

    def block_count(self) -> int:
        """当前持有的内存块数量。"""
        return len(self._blocks)
