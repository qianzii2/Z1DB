from __future__ import annotations
"""Umbra-style 16-byte inline strings — 75% of comparisons need only 4 bytes.

Paper: Neumann & Freitag, 2020 "Umbra: A Disk-Based System with In-Memory Performance"

Layout per slot (16 bytes):
  Inline (len ≤ 12): [len:1B][data:12B][pad:3B]
  Overflow (len > 12): [len:1B][prefix:4B][arena_offset:4B][arena_block:4B][pad:3B]
"""
import struct
import ctypes
from typing import Optional


class InlineStringStore:
    SLOT_SIZE = 16
    INLINE_LIMIT = 12

    __slots__ = ('_arena', '_slots', '_count', '_capacity')

    def __init__(self, capacity: int = 1024, arena: Optional[object] = None) -> None:
        from metal.arena import Arena
        self._arena: Arena = arena if arena is not None else Arena()
        self._capacity = capacity
        self._slots = bytearray(capacity * self.SLOT_SIZE)
        self._count = 0

    def append(self, s: str) -> int:
        """Append string, return index."""
        if self._count >= self._capacity:
            self._grow()

        encoded = s.encode('utf-8')
        idx = self._count
        offset = idx * self.SLOT_SIZE
        slen = len(encoded)

        if slen <= self.INLINE_LIMIT:
            # Inline: [len][data...]
            self._slots[offset] = slen
            self._slots[offset + 1: offset + 1 + slen] = encoded
            # Zero pad rest
            for i in range(offset + 1 + slen, offset + self.SLOT_SIZE):
                self._slots[i] = 0
        else:
            # Overflow: [len][prefix:4B][arena_offset:4B][data_len:4B][pad:3B]
            buf, arena_off = self._arena.alloc(slen)
            ctypes.memmove(ctypes.addressof(buf) + arena_off, encoded, slen)
            self._slots[offset] = min(slen, 255)
            # Store 4-byte prefix for fast comparison
            self._slots[offset + 1: offset + 5] = encoded[:4]
            # Store arena offset and block index
            struct.pack_into('I', self._slots, offset + 5, arena_off)
            struct.pack_into('I', self._slots, offset + 9, len(self._arena._blocks) - 1)
            # Store actual length if > 255
            if slen > 255:
                struct.pack_into('H', self._slots, offset + 13, slen)

        self._count += 1
        return idx

    def get(self, index: int) -> str:
        offset = index * self.SLOT_SIZE
        slen = self._slots[offset]

        if slen <= self.INLINE_LIMIT:
            return bytes(self._slots[offset + 1: offset + 1 + slen]).decode('utf-8')
        else:
            # Overflow: read from arena
            actual_len = slen
            if slen == 255:
                actual_len = struct.unpack_from('H', self._slots, offset + 13)[0]
            arena_off = struct.unpack_from('I', self._slots, offset + 5)[0]
            block_idx = struct.unpack_from('I', self._slots, offset + 9)[0]
            buf = self._arena._blocks[block_idx]
            raw = bytes(buf[arena_off: arena_off + actual_len])
            return raw.decode('utf-8')

    def compare(self, i: int, j: int) -> int:
        """Compare two strings. Uses 4-byte prefix for fast path.
        ~75% of comparisons resolve without touching the full string."""
        oi = i * self.SLOT_SIZE
        oj = j * self.SLOT_SIZE

        li = self._slots[oi]
        lj = self._slots[oj]

        # Compare prefixes (bytes 1-4)
        pi = self._slots[oi + 1: oi + 5]
        pj = self._slots[oj + 1: oj + 5]
        if pi != pj:
            return -1 if pi < pj else 1

        # Prefix equal — need full comparison
        si = self.get(i)
        sj = self.get(j)
        if si < sj:
            return -1
        if si > sj:
            return 1
        return 0

    def prefix_equals(self, index: int, prefix: bytes) -> bool:
        """Fast check: does string[index] start with given prefix?
        Only reads the 4-byte prefix slot if prefix ≤ 4 bytes."""
        offset = index * self.SLOT_SIZE
        plen = min(len(prefix), 4)
        stored_prefix = bytes(self._slots[offset + 1: offset + 1 + plen])
        if stored_prefix != prefix[:plen]:
            return False
        if len(prefix) <= 4:
            return True
        # Need full string check
        return self.get(index).encode('utf-8').startswith(prefix)

    def __len__(self) -> int:
        return self._count

    def _grow(self) -> None:
        new_cap = self._capacity * 2
        new_slots = bytearray(new_cap * self.SLOT_SIZE)
        new_slots[:len(self._slots)] = self._slots
        self._slots = new_slots
        self._capacity = new_cap
