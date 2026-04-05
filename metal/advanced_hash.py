from __future__ import annotations

"""Advanced hashing — Zobrist incremental, Cuckoo O(1), SipHash anti-DoS.

Zobrist: O(1) hash update when one value changes.
Cuckoo: worst-case O(1) lookup with 2 hash functions.
SipHash: DoS-resistant keyed hash (same as Python dict internals).
"""
import random
import struct
from typing import Any, Dict, List, Optional, Tuple

from metal.hash import murmur3_64


# ═══ Zobrist Incremental Hashing ═══
# Key insight: hash(row) = XOR of table[col][value] for all cols
# Update one value: hash_new = hash_old XOR table[col][old] XOR table[col][new]
# = 2 XOR operations = O(1)

class ZobristHasher:
    """O(1) hash update when a single value changes."""

    def __init__(self, num_columns: int, seed: int = 42) -> None:
        self._num_cols = num_columns
        self._rng = random.Random(seed)
        # Pre-generate random values for hash computation
        self._salt: List[int] = [self._rng.getrandbits(64) for _ in range(num_columns)]

    def hash_value(self, col: int, value: Any) -> int:
        """Hash a single column value."""
        if value is None:
            return 0
        v_hash = murmur3_64(str(value).encode('utf-8'), seed=self._salt[col])
        return v_hash

    def hash_row(self, values: list) -> int:
        """Hash a complete row."""
        h = 0
        for col, val in enumerate(values):
            h ^= self.hash_value(col, val)
        return h

    def update_hash(self, old_hash: int, col: int,
                    old_value: Any, new_value: Any) -> int:
        """O(1) update: change one column value in the hash."""
        return old_hash ^ self.hash_value(col, old_value) ^ self.hash_value(col, new_value)


# ═══ Cuckoo Hash Map ═══
# Two tables, two hash functions. Lookup = max 2 accesses (deterministic).
# Paper: Pagh & Rodler, 2004

class CuckooHashMap:
    """Deterministic O(1) lookup with 2 hash functions and 2 tables."""

    MAX_KICKS = 500

    def __init__(self, capacity: int = 256) -> None:
        self._cap = max(16, capacity)
        self._table1: List[Optional[Tuple[int, Any]]] = [None] * self._cap
        self._table2: List[Optional[Tuple[int, Any]]] = [None] * self._cap
        self._size = 0
        self._seed1 = 0
        self._seed2 = 0x9E3779B97F4A7C15

    def get(self, key: int) -> Optional[Any]:
        """O(1) worst-case lookup — at most 2 table accesses."""
        h1 = self._hash1(key)
        entry = self._table1[h1]
        if entry is not None and entry[0] == key:
            return entry[1]

        h2 = self._hash2(key)
        entry = self._table2[h2]
        if entry is not None and entry[0] == key:
            return entry[1]

        return None  # Definitely not present

    def contains(self, key: int) -> bool:
        return self.get(key) is not None

    def put(self, key: int, value: Any) -> None:
        # Check if already present
        h1 = self._hash1(key)
        if self._table1[h1] is not None and self._table1[h1][0] == key:
            self._table1[h1] = (key, value)
            return
        h2 = self._hash2(key)
        if self._table2[h2] is not None and self._table2[h2][0] == key:
            self._table2[h2] = (key, value)
            return

        # Insert with cuckoo displacement
        entry = (key, value)
        for _ in range(self.MAX_KICKS):
            h1 = self._hash1(entry[0])
            if self._table1[h1] is None:
                self._table1[h1] = entry
                self._size += 1
                return
            # Kick out existing entry
            entry, self._table1[h1] = self._table1[h1], entry

            h2 = self._hash2(entry[0])
            if self._table2[h2] is None:
                self._table2[h2] = entry
                self._size += 1
                return
            entry, self._table2[h2] = self._table2[h2], entry

        # Max kicks reached — rehash
        self._rehash()
        self.put(entry[0], entry[1])

    def remove(self, key: int) -> bool:
        h1 = self._hash1(key)
        if self._table1[h1] is not None and self._table1[h1][0] == key:
            self._table1[h1] = None
            self._size -= 1
            return True
        h2 = self._hash2(key)
        if self._table2[h2] is not None and self._table2[h2][0] == key:
            self._table2[h2] = None
            self._size -= 1
            return True
        return False

    @property
    def size(self) -> int:
        return self._size

    def _hash1(self, key: int) -> int:
        return murmur3_64(key.to_bytes(8, 'little', signed=True),
                          seed=self._seed1) % self._cap

    def _hash2(self, key: int) -> int:
        return murmur3_64(key.to_bytes(8, 'little', signed=True),
                          seed=self._seed2) % self._cap

    def _rehash(self) -> None:
        old1, old2 = self._table1, self._table2
        self._cap *= 2
        self._table1 = [None] * self._cap
        self._table2 = [None] * self._cap
        self._seed1 = random.getrandbits(64)
        self._seed2 = random.getrandbits(64)
        old_size = self._size
        self._size = 0
        for entry in old1:
            if entry is not None:
                self.put(entry[0], entry[1])
        for entry in old2:
            if entry is not None:
                self.put(entry[0], entry[1])


# ═══ SipHash-2-4 ═══
# Paper: Aumasson & Bernstein, 2012
# Used by Python dict internally. Resistant to hash-flooding attacks.

def siphash_2_4(key: bytes, k0: int = 0, k1: int = 0) -> int:
    """SipHash-2-4: keyed hash function resistant to hash-flooding.

    2 rounds per message block, 4 finalization rounds.
    """
    v0 = k0 ^ 0x736F6D6570736575
    v1 = k1 ^ 0x646F72616E646F6D
    v2 = k0 ^ 0x6C7967656E657261
    v3 = k1 ^ 0x7465646279746573

    # Pad key to multiple of 8 bytes
    msg = bytearray(key)
    msg_len = len(key)
    # Pad with zeros + length byte
    pad_len = (8 - (msg_len + 1) % 8) % 8
    msg.extend(b'\x00' * pad_len)
    msg.append(msg_len & 0xFF)
    if len(msg) % 8 != 0:
        msg.extend(b'\x00' * (8 - len(msg) % 8))

    mask64 = 0xFFFFFFFFFFFFFFFF

    def _rotl(x: int, b: int) -> int:
        return ((x << b) | (x >> (64 - b))) & mask64

    def _sipround() -> None:
        nonlocal v0, v1, v2, v3
        v0 = (v0 + v1) & mask64;
        v1 = _rotl(v1, 13);
        v1 ^= v0
        v0 = _rotl(v0, 32)
        v2 = (v2 + v3) & mask64;
        v3 = _rotl(v3, 16);
        v3 ^= v2
        v0 = (v0 + v3) & mask64;
        v3 = _rotl(v3, 21);
        v3 ^= v0
        v2 = (v2 + v1) & mask64;
        v1 = _rotl(v1, 17);
        v1 ^= v2
        v2 = _rotl(v2, 32)

    # Process 8-byte blocks
    for i in range(0, len(msg), 8):
        m = int.from_bytes(msg[i:i + 8], 'little')
        v3 ^= m
        _sipround()  # 2 rounds
        _sipround()
        v0 ^= m

    # Finalization
    v2 ^= 0xFF
    _sipround()
    _sipround()
    _sipround()
    _sipround()  # 4 rounds

    return (v0 ^ v1 ^ v2 ^ v3) & mask64


# ═══ Write-Combining Buffer ═══
# Simulates CPU write-combining — batches random writes into sequential flushes.
# Paper: Satish et al., 2010

class WriteCombiningBuffer:
    """Turns random writes into sequential flushes for cache efficiency.

    Each partition gets a small buffer (8 elements ≈ 1 cache line).
    When full, flush sequentially to output.
    """

    BUFFER_SIZE = 8  # 8 × 8 bytes = 64 bytes = 1 cache line

    def __init__(self, num_partitions: int) -> None:
        self._num_parts = num_partitions
        self._buffers: List[List[Any]] = [[] for _ in range(num_partitions)]
        self._outputs: List[List[Any]] = [[] for _ in range(num_partitions)]

    def write(self, partition: int, value: Any) -> None:
        buf = self._buffers[partition]
        buf.append(value)
        if len(buf) >= self.BUFFER_SIZE:
            self._flush(partition)

    def _flush(self, partition: int) -> None:
        self._outputs[partition].extend(self._buffers[partition])
        self._buffers[partition] = []

    def flush_all(self) -> None:
        for i in range(self._num_parts):
            if self._buffers[i]:
                self._flush(i)

    def get_partition(self, partition: int) -> List[Any]:
        self.flush_all()
        return self._outputs[partition]

    @property
    def partitions(self) -> int:
        return self._num_parts
