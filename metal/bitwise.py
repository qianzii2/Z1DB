from __future__ import annotations
"""位操作原语 — clz/ctz/popcount/2的幂次。"""


def clz64(x: int) -> int:
    """计算 64 位值的前导零数量。x=0 返回 64。"""
    if x <= 0:
        return 64 if x == 0 else 0
    n = 0
    if x <= 0x00000000FFFFFFFF:
        n += 32; x <<= 32
    if x <= 0x0000FFFFFFFFFFFF:
        n += 16; x <<= 16
    if x <= 0x00FFFFFFFFFFFFFF:
        n += 8; x <<= 8
    if x <= 0x0FFFFFFFFFFFFFFF:
        n += 4; x <<= 4
    if x <= 0x3FFFFFFFFFFFFFFF:
        n += 2; x <<= 2
    if x <= 0x7FFFFFFFFFFFFFFF:
        n += 1
    return n


def ctz64(x: int) -> int:
    """计算 64 位值的尾部零数量。x=0 返回 64。"""
    if x == 0:
        return 64
    n = 0
    x &= 0xFFFFFFFFFFFFFFFF
    if (x & 0xFFFFFFFF) == 0:
        n += 32; x >>= 32
    if (x & 0xFFFF) == 0:
        n += 16; x >>= 16
    if (x & 0xFF) == 0:
        n += 8; x >>= 8
    if (x & 0xF) == 0:
        n += 4; x >>= 4
    if (x & 0x3) == 0:
        n += 2; x >>= 2
    if (x & 0x1) == 0:
        n += 1
    return n


def popcount64(x: int) -> int:
    """64 位值的置位数（Hamming 权重）。"""
    x &= 0xFFFFFFFFFFFFFFFF
    return bin(x).count('1')


def next_power_of_2(x: int) -> int:
    """大于等于 x 的最小 2 的幂次。"""
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def is_power_of_2(x: int) -> bool:
    """判断 x 是否为 2 的幂次。"""
    return x > 0 and (x & (x - 1)) == 0
