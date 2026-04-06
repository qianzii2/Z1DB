from __future__ import annotations
"""ALP — 自适应无损浮点压缩。
论文: Afroozeh & Boncz, 2023
将浮点数分解为 10^k 倍整数 + 差分编码。
最适合科学/金融数据（如 3.14、99.99 等重复小数模式）。"""
import struct
from typing import List


def alp_encode(values: List[float]) -> bytes:
    """ALP 编码 float64 值序列。
    策略：找最优 10^k 乘数使大部分值变整数 → 差分编码。"""
    if not values:
        return struct.pack('I', 0)

    n = len(values)
    # 找最优乘数：枚举 10^0 到 10^15
    best_mult = 1
    best_int_count = 0
    for exp in range(0, 16):
        mult = 10 ** exp
        int_count = sum(
            1 for v in values[:min(100, n)]
            if abs(v * mult - round(v * mult)) < 1e-9)
        if int_count > best_int_count:
            best_int_count = int_count
            best_mult = mult

    # 转整数
    int_vals = [round(v * best_mult) for v in values]

    # 检查无损性
    lossless = all(
        abs(values[i] - int_vals[i] / best_mult) < 1e-12
        for i in range(n))

    if not lossless or best_mult == 1:
        # 回退：存原始 float64
        header = struct.pack('IBI', n, 0, 1)  # flag=0 = 原始
        data = struct.pack(f'{n}d', *values)
        return header + data

    # 差分编码整数值
    base = int_vals[0]
    deltas = [0] * n
    for i in range(1, n):
        deltas[i] = int_vals[i] - int_vals[i - 1]

    # 打包：header + base + 乘数 + zigzag varint 差分
    header = struct.pack('IBI', n, 1, best_mult)  # flag=1 = ALP
    base_bytes = struct.pack('q', base)

    delta_bytes = bytearray()
    for d in deltas:
        # Zigzag 编码有符号值
        zz = (d << 1) ^ (d >> 63) if d < 0 else d << 1
        # Varint 编码
        while zz >= 0x80:
            delta_bytes.append((zz & 0x7F) | 0x80)
            zz >>= 7
        delta_bytes.append(zz & 0x7F)

    return header + base_bytes + bytes(delta_bytes)


def alp_decode(data: bytes) -> List[float]:
    """解码 ALP 压缩数据。"""
    if len(data) < 9:
        return []

    n, flag, mult = struct.unpack_from('IBI', data, 0)
    offset = struct.calcsize('IBI')

    if flag == 0:
        # 原始 float64
        return list(struct.unpack_from(f'{n}d', data, offset))

    # ALP 编码
    base = struct.unpack_from('q', data, offset)[0]
    offset += 8

    # 解码 varint 差分
    deltas = []
    for _ in range(n):
        zz = 0
        shift = 0
        while offset < len(data):
            b = data[offset]
            offset += 1
            zz |= (b & 0x7F) << shift
            if not (b & 0x80):
                break
            shift += 7
        # Zigzag 解码
        d = -(zz >> 1) - 1 if (zz & 1) else (zz >> 1)
        deltas.append(d)

    # 重建整数值
    int_vals = [base]
    for i in range(1, len(deltas)):
        int_vals.append(int_vals[-1] + deltas[i])
    while len(int_vals) < n:
        int_vals.append(int_vals[-1] if int_vals else 0)

    return [v / mult for v in int_vals]


def alp_compression_ratio(values: List[float]) -> float:
    if not values:
        return 1.0
    encoded = alp_encode(values)
    original = len(values) * 8
    return len(encoded) / original
