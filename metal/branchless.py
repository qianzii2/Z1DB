from __future__ import annotations
"""无分支算术 — 单值函数保留作为教学展示。
注：Python解释器中这些优化无实际性能收益，但展示了底层原理。"""


def min_int(a: int, b: int) -> int:
    """无分支min（64位有符号整数）。"""
    diff = a - b
    return b + (diff & (diff >> 63))


def max_int(a: int, b: int) -> int:
    """无分支max（64位有符号整数）。"""
    diff = a - b
    return a - (diff & (diff >> 63))


def abs_int(x: int) -> int:
    """无分支绝对值。"""
    mask = x >> 63
    return (x ^ mask) - mask


def clamp(x: int, lo: int, hi: int) -> int:
    """钳制到[lo, hi]范围。"""
    return max_int(lo, min_int(x, hi))


def sign(x: int) -> int:
    """符号函数：-1, 0, 1。"""
    return (x > 0) - (x < 0)


def conditional_select(cond: bool, a: int, b: int) -> int:
    """CMOV模拟：cond为True返回a，否则返回b。"""
    mask = -(int(cond))
    return (a & mask) | (b & ~mask)
