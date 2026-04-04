from __future__ import annotations
"""Hash functions: murmur3, fibonacci, combiners."""

from metal.config import NULL_HASH_SENTINEL


def _to_bytes(key: bytes) -> bytes:
    if isinstance(key, (bytearray, memoryview)):
        return bytes(key)
    return key


def murmur3_64(key: bytes, seed: int = 0) -> int:
    """MurmurHash3 finalizer-style 64-bit hash."""
    key = _to_bytes(key)
    h = seed & 0xFFFFFFFFFFFFFFFF
    for b in key:
        h ^= b
        h = (h * 0x5bd1e9955bd1e995) & 0xFFFFFFFFFFFFFFFF
        h ^= h >> 47
    h = (h * 0x5bd1e9955bd1e995) & 0xFFFFFFFFFFFFFFFF
    h ^= h >> 47
    return h


def murmur3_128(key: bytes, seed: int = 0) -> tuple[int, int]:
    h1 = murmur3_64(key, seed)
    h2 = murmur3_64(key, seed ^ 0x9E3779B97F4A7C15)
    return (h1, h2)


def fibonacci_hash(key: int, shift: int) -> int:
    return ((key * 11400714819323198485) & 0xFFFFFFFFFFFFFFFF) >> shift


def hash_combine(h1: int, h2: int) -> int:
    h1 ^= h2 + 0x9E3779B97F4A7C15 + (h1 << 6) + (h1 >> 2)
    return h1 & 0xFFFFFFFFFFFFFFFF


def hash_value(val: object, dtype_code: str) -> int:
    """Hash a typed value. None → NULL_HASH_SENTINEL."""
    if val is None:
        return NULL_HASH_SENTINEL
    if isinstance(val, bool):
        return murmur3_64(b'\x01' if val else b'\x00')
    if isinstance(val, int):
        return murmur3_64(val.to_bytes(8, 'little', signed=True))
    if isinstance(val, float):
        import struct
        return murmur3_64(struct.pack('<d', val))
    if isinstance(val, str):
        return murmur3_64(val.encode('utf-8'))
    return murmur3_64(str(val).encode('utf-8'))
