from __future__ import annotations
"""哈希函数：高质量blake2b替代伪MurmurHash，fibonacci，combiners。"""
import hashlib
import struct
from metal.config import NULL_HASH_SENTINEL


def murmur3_64(key: bytes, seed: int = 0) -> int:
    """高质量64位哈希。使用blake2b（比手写逐字节MurmurHash快且分布更好）。"""
    if isinstance(key, (bytearray, memoryview)):
        key = bytes(key)
    seed_bytes = (seed & 0xFFFFFFFFFFFFFFFF).to_bytes(8, 'little')
    h = hashlib.blake2b(key, digest_size=8, key=seed_bytes).digest()
    return int.from_bytes(h, 'little')


def murmur3_128(key: bytes, seed: int = 0) -> tuple[int, int]:
    """128位哈希，返回(h1, h2)。"""
    if isinstance(key, (bytearray, memoryview)):
        key = bytes(key)
    seed_bytes = (seed & 0xFFFFFFFFFFFFFFFF).to_bytes(8, 'little')
    h = hashlib.blake2b(key, digest_size=16, key=seed_bytes).digest()
    return (int.from_bytes(h[:8], 'little'), int.from_bytes(h[8:], 'little'))


def fibonacci_hash(key: int, shift: int) -> int:
    """Fibonacci哈希（乘法哈希）。"""
    return ((key * 11400714819323198485) & 0xFFFFFFFFFFFFFFFF) >> shift


def hash_combine(h1: int, h2: int) -> int:
    """组合两个哈希值。"""
    h1 ^= h2 + 0x9E3779B97F4A7C15 + (h1 << 6) + (h1 >> 2)
    return h1 & 0xFFFFFFFFFFFFFFFF


def hash_value(val: object, dtype_code: str = '') -> int:
    """对类型值做哈希。None → NULL_HASH_SENTINEL。"""
    if val is None:
        return NULL_HASH_SENTINEL
    if isinstance(val, bool):
        return murmur3_64(b'\x01' if val else b'\x00')
    if isinstance(val, int):
        return murmur3_64(val.to_bytes(8, 'little', signed=True))
    if isinstance(val, float):
        return murmur3_64(struct.pack('<d', val))
    if isinstance(val, str):
        return murmur3_64(val.encode('utf-8'))
    if isinstance(val, bytes):
        return murmur3_64(val)
    return murmur3_64(str(val).encode('utf-8'))
