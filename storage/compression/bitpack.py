from __future__ import annotations
"""位打包 (BitPacking) — 用最小位宽存储整数。
示例：值域 [0,15] → 每值 4 位 → 相比 int64 压缩 16x。
FOR (Frame of Reference)：减去最小值后位打包。"""
from typing import List, Tuple


def bitpack_encode(values: list,
                   bit_width: int) -> Tuple[bytes, int]:
    """按 bit_width 位打包。返回 (packed_bytes, count)。"""
    if not values or bit_width <= 0:
        return b'', 0
    total_bits = len(values) * bit_width
    buf = bytearray((total_bits + 7) // 8)
    bit_pos = 0
    for v in values:
        val = v & ((1 << bit_width) - 1)
        for b in range(bit_width):
            if val & (1 << b):
                byte_idx = bit_pos >> 3
                bit_idx = bit_pos & 7
                buf[byte_idx] |= (1 << bit_idx)
            bit_pos += 1
    return bytes(buf), len(values)


def bitpack_decode(data: bytes, bit_width: int,
                   count: int) -> list:
    """解包。"""
    result = []
    bit_pos = 0
    for _ in range(count):
        val = 0
        for b in range(bit_width):
            byte_idx = bit_pos >> 3
            bit_idx = bit_pos & 7
            if (byte_idx < len(data)
                    and data[byte_idx] & (1 << bit_idx)):
                val |= (1 << b)
            bit_pos += 1
        result.append(val)
    return result


def for_encode(values: list) -> Tuple[int, bytes, int, int]:
    """FOR 编码：减去最小值后位打包。
    返回 (min_val, packed_bytes, count, bit_width)。"""
    if not values:
        return 0, b'', 0, 0
    min_val = min(values)
    residuals = [v - min_val for v in values]
    max_residual = max(residuals) if residuals else 0
    bit_width = max(1, max_residual.bit_length())
    packed, count = bitpack_encode(residuals, bit_width)
    return min_val, packed, count, bit_width


def for_decode(min_val: int, packed: bytes,
               count: int, bit_width: int) -> list:
    """FOR 解码。"""
    residuals = bitpack_decode(packed, bit_width, count)
    return [r + min_val for r in residuals]


def for_compression_ratio(values: list) -> float:
    """估算 FOR 压缩率。"""
    if not values:
        return 1.0
    min_val = min(values)
    max_val = max(values)
    range_val = max_val - min_val
    bit_width = max(1, range_val.bit_length()) if range_val > 0 else 1
    return (len(values) * bit_width) / (len(values) * 64)
