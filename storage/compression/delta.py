from __future__ import annotations
"""差分编码 — 最适合有序/递增整数列（ID、时间戳）。
存储相邻值的差值，差值通常很小，后续可用 BitPacking 进一步压缩。"""
from typing import List, Tuple


def delta_encode(values: list) -> Tuple[int, list]:
    """编码为 (base, deltas)。base = 首值，deltas = 差值序列。"""
    if not values:
        return 0, []
    base = values[0]
    deltas = [0] * len(values)
    deltas[0] = 0
    for i in range(1, len(values)):
        deltas[i] = values[i] - values[i - 1]
    return base, deltas


def delta_decode(base: int, deltas: list) -> list:
    """解码回原始值。"""
    if not deltas:
        return []
    result = [base + deltas[0]]
    for i in range(1, len(deltas)):
        result.append(result[-1] + deltas[i])
    return result


def delta_of_delta_encode(values: list) -> Tuple[int, int, list]:
    """二阶差分：最适合近等步长序列（固定间隔时间戳等）。
    二阶差分 ≈ 0 时压缩率极高。"""
    if len(values) < 2:
        return values[0] if values else 0, 0, []
    base = values[0]
    first_delta = values[1] - values[0]
    dod = [0] * len(values)
    prev_delta = first_delta
    for i in range(2, len(values)):
        curr_delta = values[i] - values[i - 1]
        dod[i] = curr_delta - prev_delta
        prev_delta = curr_delta
    return base, first_delta, dod


def delta_of_delta_decode(base: int, first_delta: int,
                          dod: list) -> list:
    if not dod:
        return [base]
    result = [base, base + first_delta]
    delta = first_delta
    for i in range(2, len(dod)):
        delta += dod[i]
        result.append(result[-1] + delta)
    return result


def max_delta(deltas: list) -> int:
    """最大绝对差值 — 决定 BitPacking 所需位宽。"""
    if not deltas:
        return 0
    return max(abs(d) for d in deltas)


def bits_needed(max_val: int) -> int:
    """表示一个值所需的最小位数。"""
    if max_val <= 0:
        return 1
    return max_val.bit_length() + 1  # +1 用于符号位
