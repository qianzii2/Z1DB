from __future__ import annotations
"""Radix Sort — O(n*w)。
radix_sort_int64: 有符号 64 位整数排序。
radix_sort_uint: 无符号整数排序（用于字典编码 code 排序）。"""
import array as _array
from typing import List, Optional, Tuple


def radix_sort_int64(values: list, indices: Optional[list] = None,
                     descending: bool = False) -> Tuple[list, list]:
    """LSD 基数排序有符号 int64。8 趟 × 256 桶。
    返回 (sorted_values, sorted_indices)。"""
    n = len(values)
    if n <= 1:
        idx = indices if indices is not None else list(range(n))
        return list(values), list(idx)
    if indices is None:
        indices = list(range(n))

    # 翻转符号位：有符号 → 可正确排序的无符号
    uvals = _array.array('Q', [0] * n)
    for i in range(n):
        uvals[i] = values[i] ^ (1 << 63)

    src_vals = uvals
    src_idx = _array.array('Q', indices)
    dst_vals = _array.array('Q', [0] * n)
    dst_idx = _array.array('Q', [0] * n)

    for byte_num in range(8):
        shift = byte_num * 8
        counts = [0] * 256
        for i in range(n):
            counts[(src_vals[i] >> shift) & 0xFF] += 1
        if descending:
            total = 0
            for i in range(255, -1, -1):
                old = counts[i]; counts[i] = total; total += old
        else:
            total = 0
            for i in range(256):
                old = counts[i]; counts[i] = total; total += old
        for i in range(n):
            b = (src_vals[i] >> shift) & 0xFF
            pos = counts[b]
            dst_vals[pos] = src_vals[i]
            dst_idx[pos] = src_idx[i]
            counts[b] += 1
        src_vals, dst_vals = dst_vals, src_vals
        src_idx, dst_idx = dst_idx, src_idx

    # 还原有符号
    result_vals = [int(v) ^ (1 << 63) for v in src_vals]
    result_vals = [v - (1 << 64) if v >= (1 << 63) else v
                   for v in result_vals]
    result_idx = [int(v) for v in src_idx]
    return result_vals, result_idx


def radix_sort_uint(values: list, bit_width: int = 32,
                    indices: Optional[list] = None,
                    descending: bool = False) -> Tuple[list, list]:
    """LSD 基数排序无符号整数。
    用于字典编码 code 排序、bitmap 索引等。
    返回 (sorted_values, sorted_indices)。"""
    n = len(values)
    if indices is None:
        indices = list(range(n))
    if n <= 1:
        return list(values), list(indices)

    src_vals = list(values)
    src_idx = list(indices)
    dst_vals = [0] * n
    dst_idx = [0] * n
    passes = (bit_width + 7) // 8

    for byte_num in range(passes):
        shift = byte_num * 8
        counts = [0] * 256
        for v in src_vals:
            counts[(v >> shift) & 0xFF] += 1
        if descending:
            total = 0
            for i in range(255, -1, -1):
                old = counts[i]; counts[i] = total; total += old
        else:
            total = 0
            for i in range(256):
                old = counts[i]; counts[i] = total; total += old
        for i in range(n):
            b = (src_vals[i] >> shift) & 0xFF
            pos = counts[b]
            dst_vals[pos] = src_vals[i]
            dst_idx[pos] = src_idx[i]
            counts[b] += 1
        src_vals, dst_vals = dst_vals, src_vals
        src_idx, dst_idx = dst_idx, src_idx

    return src_vals, src_idx
