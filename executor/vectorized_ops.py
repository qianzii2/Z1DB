from __future__ import annotations
"""向量化原语 — 整列操作，减少Python解释器开销。
提供batch级别的算术/比较/聚合操作。"""
import array as _array
from typing import Any, Callable, List, Optional, Tuple


# ═══ 向量算术 ═══

def vec_add_fast(a: _array.array, b: _array.array, n: int) -> _array.array:
    """无NULL的向量加法。直接操作array.array。"""
    code = a.typecode
    out = _array.array(code, [a[i] + b[i] for i in range(n)])
    return out


def vec_sub_fast(a: _array.array, b: _array.array, n: int) -> _array.array:
    code = a.typecode
    return _array.array(code, [a[i] - b[i] for i in range(n)])


def vec_mul_fast(a: _array.array, b: _array.array, n: int) -> _array.array:
    code = a.typecode
    return _array.array(code, [a[i] * b[i] for i in range(n)])


# ═══ 向量比较 ═══

def vec_cmp_eq_fast(a: _array.array, b: _array.array,
                    out_bitmap: bytearray, n: int) -> int:
    """无NULL的等值比较。返回匹配数。"""
    count = 0
    for i in range(n):
        if a[i] == b[i]:
            out_bitmap[i >> 3] |= (1 << (i & 7))
            count += 1
    return count


def vec_cmp_scalar_eq(data: _array.array, target: Any,
                      out_bitmap: bytearray, n: int) -> int:
    """列 vs 标量等值比较。"""
    count = 0
    for i in range(n):
        if data[i] == target:
            out_bitmap[i >> 3] |= (1 << (i & 7))
            count += 1
    return count


def vec_cmp_scalar_gt(data: _array.array, threshold: Any,
                      out_bitmap: bytearray, n: int) -> int:
    count = 0
    for i in range(n):
        if data[i] > threshold:
            out_bitmap[i >> 3] |= (1 << (i & 7))
            count += 1
    return count


def vec_cmp_scalar_lt(data: _array.array, threshold: Any,
                      out_bitmap: bytearray, n: int) -> int:
    count = 0
    for i in range(n):
        if data[i] < threshold:
            out_bitmap[i >> 3] |= (1 << (i & 7))
            count += 1
    return count


def vec_cmp_scalar_gte(data: _array.array, threshold: Any,
                       out_bitmap: bytearray, n: int) -> int:
    count = 0
    for i in range(n):
        if data[i] >= threshold:
            out_bitmap[i >> 3] |= (1 << (i & 7))
            count += 1
    return count


def vec_cmp_scalar_lte(data: _array.array, threshold: Any,
                       out_bitmap: bytearray, n: int) -> int:
    count = 0
    for i in range(n):
        if data[i] <= threshold:
            out_bitmap[i >> 3] |= (1 << (i & 7))
            count += 1
    return count


# ═══ 向量聚合 ═══

def vec_sum_fast(data: _array.array, n: int) -> Any:
    """无NULL求和。"""
    return sum(data[:n])


def vec_min_max_fast(data: _array.array, n: int) -> Tuple[Any, Any]:
    """无NULL求最值。"""
    if n == 0:
        return None, None
    mn = mx = data[0]
    for i in range(1, n):
        v = data[i]
        if v < mn: mn = v
        if v > mx: mx = v
    return mn, mx


def vec_count_nonnull(null_bitmap: bytearray, n: int) -> int:
    """计算非NULL行数。"""
    null_count = 0
    for i in range(n):
        if (null_bitmap[i >> 3] >> (i & 7)) & 1:
            null_count += 1
    return n - null_count


# ═══ evaluator集成入口 ═══

def try_vectorized_arith(op: str, left_arr: _array.array,
                         right_arr: _array.array, n: int,
                         typecode: str) -> Optional[_array.array]:
    """尝试向量化算术。成功返回结果array，失败返回None。"""
    try:
        if op == '+': return vec_add_fast(left_arr, right_arr, n)
        if op == '-': return vec_sub_fast(left_arr, right_arr, n)
        if op == '*': return vec_mul_fast(left_arr, right_arr, n)
    except (OverflowError, TypeError):
        return None
    return None


def try_vectorized_cmp_scalar(op: str, data: _array.array,
                              target: Any, n: int) -> Optional[bytearray]:
    """尝试向量化标量比较。返回结果bitmap或None。"""
    bmp = bytearray((n + 7) // 8)
    try:
        if op == '=': vec_cmp_scalar_eq(data, target, bmp, n)
        elif op == '>': vec_cmp_scalar_gt(data, target, bmp, n)
        elif op == '<': vec_cmp_scalar_lt(data, target, bmp, n)
        elif op == '>=': vec_cmp_scalar_gte(data, target, bmp, n)
        elif op == '<=': vec_cmp_scalar_lte(data, target, bmp, n)
        else: return None
        return bmp
    except TypeError:
        return None
