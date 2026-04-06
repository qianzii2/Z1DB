from __future__ import annotations
"""Z1DB 哈希函数 — SipHash-2-4 主实现。
SipHash 抗 HashDoS，适合哈希表键计算。
z1hash64：64 位键控哈希。
z1hash128：128 位版本（双重哈希，Bloom Filter 用）。"""
import struct
from metal.config import NULL_HASH_SENTINEL

_MASK64 = 0xFFFFFFFFFFFFFFFF

# 预编译 struct 对象，避免每次调用解析格式字符串 [P07]
_STRUCT_D = struct.Struct('<d')
_STRUCT_Q = struct.Struct('<q')


def _rotl(x: int, b: int) -> int:
    """64 位循环左移。"""
    return ((x << b) | (x >> (64 - b))) & _MASK64


def _siphash_round(v0, v1, v2, v3):
    """SipHash 单轮压缩。"""
    v0 = (v0 + v1) & _MASK64; v1 = _rotl(v1, 13)
    v1 ^= v0; v0 = _rotl(v0, 32)
    v2 = (v2 + v3) & _MASK64; v3 = _rotl(v3, 16)
    v3 ^= v2
    v0 = (v0 + v3) & _MASK64; v3 = _rotl(v3, 21)
    v3 ^= v0
    v2 = (v2 + v1) & _MASK64; v1 = _rotl(v1, 17)
    v1 ^= v2; v2 = _rotl(v2, 32)
    return v0, v1, v2, v3


def _siphash_2_4(key: bytes, k0: int = 0,
                 k1: int = 0) -> int:
    """SipHash-2-4：2 轮消息处理 + 4 轮终结。"""
    v0 = k0 ^ 0x736F6D6570736575
    v1 = k1 ^ 0x646F72616E646F6D
    v2 = k0 ^ 0x6C7967656E657261
    v3 = k1 ^ 0x7465646279746573

    msg_len = len(key)
    blocks = msg_len // 8
    for i in range(blocks):
        m = int.from_bytes(key[i * 8:(i + 1) * 8], 'little')
        v3 ^= m
        # 2 轮压缩
        v0, v1, v2, v3 = _siphash_round(v0, v1, v2, v3)
        v0, v1, v2, v3 = _siphash_round(v0, v1, v2, v3)
        v0 ^= m

    # 最后一块（含长度字节）
    last = bytearray(8)
    tail_start = blocks * 8
    for i in range(msg_len - tail_start):
        last[i] = key[tail_start + i]
    last[7] = msg_len & 0xFF
    m = int.from_bytes(last, 'little')
    v3 ^= m
    v0, v1, v2, v3 = _siphash_round(v0, v1, v2, v3)
    v0, v1, v2, v3 = _siphash_round(v0, v1, v2, v3)
    v0 ^= m

    # 4 轮终结
    v2 ^= 0xFF
    for _ in range(4):
        v0, v1, v2, v3 = _siphash_round(v0, v1, v2, v3)

    return (v0 ^ v1 ^ v2 ^ v3) & _MASK64


# ═══ 公开 API ═══

def z1hash64(key: bytes, seed: int = 0) -> int:
    """64 位 SipHash。seed 作为 k0，k1 从 seed 派生。"""
    if isinstance(key, (bytearray, memoryview)):
        key = bytes(key)
    k0 = seed & _MASK64
    k1 = ((seed * 0x9E3779B97F4A7C15) + 1) & _MASK64
    return _siphash_2_4(key, k0, k1)


def z1hash128(key: bytes, seed: int = 0) -> tuple[int, int]:
    """128 位哈希（Bloom Filter 双重哈希用）。返回 (h1, h2)。"""
    if isinstance(key, (bytearray, memoryview)):
        key = bytes(key)
    k0 = seed & _MASK64
    k1 = ((seed * 0x9E3779B97F4A7C15) + 1) & _MASK64
    h1 = _siphash_2_4(key, k0, k1)
    h2 = _siphash_2_4(key, k0 ^ 0xDEADBEEF, k1 ^ 0xCAFEBABE)
    return h1, h2


# 兼容旧调用方（已废弃，请用 z1hash64）
murmur3_64 = z1hash64


def fibonacci_hash(key: int, shift: int) -> int:
    """Fibonacci 哈希（黄金比例乘法哈希）。用于开放寻址表。"""
    return ((key * 11400714819323198485)
            & _MASK64) >> shift


def hash_combine(h1: int, h2: int) -> int:
    """组合两个哈希值（Boost hash_combine 变体）。"""
    h1 ^= (h2 + 0x9E3779B97F4A7C15
           + (h1 << 6) + (h1 >> 2))
    return h1 & _MASK64


def hash_value(val: object, dtype_code: str = '') -> int:
    """通用值哈希。None → NULL_HASH_SENTINEL。
    使用预编译 struct 对象避免格式字符串解析开销 [P07]。"""
    if val is None:
        return NULL_HASH_SENTINEL
    if isinstance(val, bool):
        return z1hash64(b'\x01' if val else b'\x00')
    if isinstance(val, int):
        return z1hash64(
            val.to_bytes(8, 'little', signed=True))
    if isinstance(val, float):
        return z1hash64(_STRUCT_D.pack(val))
    if isinstance(val, str):
        return z1hash64(val.encode('utf-8'))
    if isinstance(val, bytes):
        return z1hash64(val)
    return z1hash64(str(val).encode('utf-8'))
