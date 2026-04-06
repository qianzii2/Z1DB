from __future__ import annotations
"""Umbra 风格 16 字节内联字符串存储。
论文: Neumann & Freitag, 2020 "Umbra: A Disk-Based System with In-Memory Performance"

约 75% 的字符串比较只需 4 字节前缀即可判断大小关系。

每个 slot 16 字节布局:
  内联模式 (len ≤ 12):
    [len:1B][data:12B][pad:3B]
    数据直接存在 slot 内，无额外内存分配。

  溢出模式 (len > 12):
    [255:1B][prefix:4B][arena_off:4B][block_idx:4B][actual_len:2B][pad:1B]
    前 4 字节前缀用于快速比较，完整数据存在 Arena 中。
"""
import struct
from typing import Optional
from utils.errors import ExecutionError


class InlineStringStore:
    """固定 slot 的字符串列存储。
    特点：
      - O(1) 随机访问
      - 75% 比较在 4 字节前缀阶段完成
      - 短字符串零额外分配（内联在 slot 中）
      - 长字符串溢出到 Arena（批量释放）
    """

    SLOT_SIZE = 16
    INLINE_LIMIT = 12

    __slots__ = ('_arena', '_slots', '_count', '_capacity')

    def __init__(self, capacity: int = 1024,
                 arena: Optional[object] = None) -> None:
        from metal.arena import Arena
        self._arena: Arena = arena if arena is not None else Arena()
        self._capacity = capacity
        self._slots = bytearray(capacity * self.SLOT_SIZE)
        self._count = 0

    def append(self, s: str) -> int:
        """追加字符串，返回 slot 索引。"""
        if self._count >= self._capacity:
            self._grow()

        encoded = s.encode('utf-8')
        slen = len(encoded)
        slot_idx = self._count
        offset = slot_idx * self.SLOT_SIZE

        if slen <= self.INLINE_LIMIT:
            # 内联模式: [len:1B][data:最多12B][pad]
            self._slots[offset] = slen
            self._slots[offset + 1: offset + 1 + slen] = encoded
            # 剩余字节清零
            for i in range(offset + 1 + slen,
                           offset + self.SLOT_SIZE):
                self._slots[i] = 0
        else:
            # 溢出模式
            if slen > 65535:
                raise ExecutionError(
                    f"字符串过长: {slen} > 65535 字节")

            # 写入 Arena
            import ctypes
            buf, arena_offset = self._arena.alloc(slen)
            ctypes.memmove(
                ctypes.addressof(buf) + arena_offset,
                encoded, slen)
            block_idx = len(self._arena._blocks) - 1

            self._slots[offset] = 255  # 溢出标记
            # 前缀（前 4 字节，用于快速比较）
            prefix = encoded[:4]
            self._slots[offset + 1: offset + 1 + len(prefix)] = prefix
            for i in range(len(prefix), 4):
                self._slots[offset + 1 + i] = 0
            # arena 偏移 (4B)
            struct.pack_into('<I', self._slots,
                             offset + 5, arena_offset)
            # block 索引 (4B)
            struct.pack_into('<I', self._slots,
                             offset + 9, block_idx)
            # 真实长度 (2B)
            struct.pack_into('<H', self._slots,
                             offset + 13, slen)
            # pad (1B)
            self._slots[offset + 15] = 0

        self._count += 1
        return slot_idx

    def get(self, index: int) -> str:
        """按 slot 索引获取字符串。O(1)。"""
        if index < 0 or index >= self._count:
            raise ExecutionError(
                f"InlineStringStore 索引越界: "
                f"{index}, 总数: {self._count}")

        offset = index * self.SLOT_SIZE
        slen = self._slots[offset]

        if slen <= self.INLINE_LIMIT:
            # 内联模式：直接从 slot 读取
            return bytes(
                self._slots[offset + 1: offset + 1 + slen]
            ).decode('utf-8')
        else:
            # 溢出模式：从 Arena 读取
            arena_off = struct.unpack_from(
                '<I', self._slots, offset + 5)[0]
            block_idx = struct.unpack_from(
                '<I', self._slots, offset + 9)[0]
            actual_len = struct.unpack_from(
                '<H', self._slots, offset + 13)[0]

            if block_idx >= len(self._arena._blocks):
                raise ExecutionError(
                    f"InlineStringStore: "
                    f"无效 block 索引 {block_idx}")

            buf = self._arena._blocks[block_idx]
            raw = bytes(
                buf[arena_off: arena_off + actual_len])
            return raw.decode('utf-8')

    def compare(self, i: int, j: int) -> int:
        """比较两个字符串。
        先用 4 字节前缀快速判断（约 75% 在此阶段完成），
        前缀相同再全量比较。"""
        oi = i * self.SLOT_SIZE
        oj = j * self.SLOT_SIZE

        # 前缀比较（bytes 1-4）
        pi = self._slots[oi + 1: oi + 5]
        pj = self._slots[oj + 1: oj + 5]
        if pi != pj:
            return -1 if pi < pj else 1

        # 前缀相同 → 全量比较
        si = self.get(i)
        sj = self.get(j)
        if si < sj:
            return -1
        if si > sj:
            return 1
        return 0

    def prefix_equals(self, index: int,
                      prefix: bytes) -> bool:
        """快速前缀匹配。prefix ≤ 4 字节时只读 slot 内的前缀字段。"""
        offset = index * self.SLOT_SIZE
        plen = min(len(prefix), 4)
        stored_prefix = bytes(
            self._slots[offset + 1: offset + 1 + plen])
        if stored_prefix != prefix[:plen]:
            return False
        if len(prefix) <= 4:
            return True
        # prefix > 4 字节：需要全量检查
        return self.get(index).encode('utf-8').startswith(
            prefix)

    def __len__(self) -> int:
        return self._count

    def _grow(self) -> None:
        """容量翻倍。"""
        new_cap = self._capacity * 2
        new_slots = bytearray(new_cap * self.SLOT_SIZE)
        new_slots[:len(self._slots)] = self._slots
        self._slots = new_slots
        self._capacity = new_cap
