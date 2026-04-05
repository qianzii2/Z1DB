from __future__ import annotations

"""ALP — Adaptive Lossless floating Point compression.
Paper: Afroozeh & Boncz, 2023.
Decomposes floats into exponent + mantissa, encodes separately.
Best for scientific/financial data with repeated decimal patterns."""
import struct
from typing import List, Tuple


def alp_encode(values: List[float]) -> bytes:
    """Encode float64 values using ALP.

    Strategy:
    1. Find common decimal multiplier (e.g., values like 3.14 → multiply by 100)
    2. Convert to integers
    3. Delta-encode the integers
    4. Store multiplier + deltas
    """
    if not values:
        return struct.pack('I', 0)

    n = len(values)
    # Find best multiplier (power of 10 that makes most values integers)
    best_mult = 1
    best_int_count = 0
    for exp in range(0, 16):
        mult = 10 ** exp
        int_count = sum(1 for v in values[:min(100, n)]
                        if abs(v * mult - round(v * mult)) < 1e-9)
        if int_count > best_int_count:
            best_int_count = int_count
            best_mult = mult

    # Convert to integers
    int_vals = [round(v * best_mult) for v in values]

    # Check if all conversions are lossless
    lossless = all(abs(values[i] - int_vals[i] / best_mult) < 1e-12
                   for i in range(n))

    if not lossless or best_mult == 1:
        # Fallback: store raw float64
        header = struct.pack('IBI', n, 0, 1)  # flag=0 means raw
        data = struct.pack(f'{n}d', *values)
        return header + data

    # Delta encode the integer values
    base = int_vals[0]
    deltas = [0] * n
    for i in range(1, n):
        deltas[i] = int_vals[i] - int_vals[i - 1]

    # Pack: header + base + multiplier + deltas
    header = struct.pack('IBI', n, 1, best_mult)  # flag=1 means ALP
    base_bytes = struct.pack('q', base)

    # Variable-length encode deltas
    delta_bytes = bytearray()
    for d in deltas:
        # Zigzag encoding for signed values
        zz = (d << 1) ^ (d >> 63) if d < 0 else d << 1
        # Varint encoding
        while zz >= 0x80:
            delta_bytes.append((zz & 0x7F) | 0x80)
            zz >>= 7
        delta_bytes.append(zz & 0x7F)

    return header + base_bytes + bytes(delta_bytes)


def alp_decode(data: bytes) -> List[float]:
    """Decode ALP-compressed float64 values."""
    if len(data) < 9:
        return []

    n, flag, mult = struct.unpack_from('IBI', data, 0)
    offset = struct.calcsize('IBI')

    if flag == 0:
        # Raw float64
        return list(struct.unpack_from(f'{n}d', data, offset))

    # ALP encoded
    base = struct.unpack_from('q', data, offset)[0]
    offset += 8

    # Decode varint deltas
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
        # Zigzag decode
        d = -(zz >> 1) - 1 if (zz & 1) else (zz >> 1)
        deltas.append(d)

    # Reconstruct
    int_vals = [base]
    for i in range(1, len(deltas)):
        int_vals.append(int_vals[-1] + deltas[i])
    while len(int_vals) < n:
        int_vals.append(int_vals[-1] if int_vals else 0)

    return [v / mult for v in int_vals]


def alp_compression_ratio(values: List[float]) -> float:
    if not values: return 1.0
    encoded = alp_encode(values)
    original = len(values) * 8
    return len(encoded) / original
