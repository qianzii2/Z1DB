from __future__ import annotations
"""Gorilla compression for floats. Paper: Pelkonen et al., 2015 (Facebook).
XOR adjacent floats, store only the changed bits. ~12x compression for timeseries."""
import struct
from typing import List, Tuple


def gorilla_encode(values: list) -> bytes:
    """Encode float64 values using Gorilla XOR compression."""
    if not values:
        return b''
    bits: list = []
    prev = struct.unpack('Q', struct.pack('d', float(values[0])))[0]
    # First value: store all 64 bits
    for i in range(63, -1, -1):
        bits.append((prev >> i) & 1)
    for idx in range(1, len(values)):
        curr = struct.unpack('Q', struct.pack('d', float(values[idx])))[0]
        xor = prev ^ curr
        if xor == 0:
            bits.append(0)  # Same as previous: 1 bit
        else:
            bits.append(1)
            leading = _clz64(xor)
            trailing = _ctz64(xor)
            sig_bits = 64 - leading - trailing
            if sig_bits <= 0:
                sig_bits = 1
            # Store leading zeros (6 bits) + significant bits length (6 bits) + significant bits
            bits.append(1)
            for i in range(5, -1, -1):
                bits.append((leading >> i) & 1)
            for i in range(5, -1, -1):
                bits.append((sig_bits >> i) & 1)
            for i in range(sig_bits - 1, -1, -1):
                bits.append((xor >> (trailing + i)) & 1)
        prev = curr
    # Pack bits into bytes
    result = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            if i + j < len(bits):
                byte |= (bits[i + j] << (7 - j))
        result.append(byte)
    # Prepend count
    return struct.pack('I', len(values)) + bytes(result)


def gorilla_decode(data: bytes) -> list:
    """Decode Gorilla-compressed float64 values."""
    if len(data) < 4:
        return []
    count = struct.unpack('I', data[:4])[0]
    if count == 0:
        return []
    bits_data = data[4:]
    bits: list = []
    for byte in bits_data:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    pos = 0

    def read_bits(n: int) -> int:
        nonlocal pos
        val = 0
        for _ in range(n):
            if pos < len(bits):
                val = (val << 1) | bits[pos]
                pos += 1
        return val

    # First value: 64 bits
    first_raw = read_bits(64)
    values = [struct.unpack('d', struct.pack('Q', first_raw))[0]]
    prev = first_raw
    for _ in range(1, count):
        if pos >= len(bits):
            break
        flag = read_bits(1)
        if flag == 0:
            values.append(struct.unpack('d', struct.pack('Q', prev))[0])
        else:
            ctrl = read_bits(1)
            if ctrl == 1:
                leading = read_bits(6)
                sig_bits = read_bits(6)
                if sig_bits == 0:
                    sig_bits = 64
                sig_val = read_bits(sig_bits)
                trailing = 64 - leading - sig_bits
                xor = sig_val << trailing
            else:
                xor = read_bits(64)
            curr = prev ^ xor
            values.append(struct.unpack('d', struct.pack('Q', curr))[0])
            prev = curr
    return values


def gorilla_compression_ratio(values: list) -> float:
    """Estimate compression ratio."""
    if not values:
        return 1.0
    encoded = gorilla_encode(values)
    original = len(values) * 8
    return len(encoded) / original if original > 0 else 1.0


def _clz64(x: int) -> int:
    if x == 0:
        return 64
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


def _ctz64(x: int) -> int:
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
