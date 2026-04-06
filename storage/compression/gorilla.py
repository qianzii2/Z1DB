from __future__ import annotations
"""Gorilla 压缩 — 浮点数 XOR 编码。
论文: Pelkonen et al., 2015 "Gorilla: A Fast, Scalable, In-Memory Time Series Database"
XOR 相邻 float64 值，仅存储变化的位。时序数据压缩率可达 12x。"""
import struct
from typing import List
from metal.bitwise import clz64, ctz64


def gorilla_encode(values: list) -> bytes:
    """Gorilla XOR 编码 float64 值序列。"""
    if not values:
        return b''
    bits: list = []
    prev = struct.unpack('Q', struct.pack('d', float(values[0])))[0]
    # 首值：存全部 64 位
    for i in range(63, -1, -1):
        bits.append((prev >> i) & 1)
    for idx in range(1, len(values)):
        curr = struct.unpack('Q', struct.pack('d', float(values[idx])))[0]
        xor = prev ^ curr
        if xor == 0:
            # 与前值相同：1 位标记
            bits.append(0)
        else:
            bits.append(1)
            leading = clz64(xor)
            trailing = ctz64(xor)
            sig_bits = 64 - leading - trailing
            if sig_bits <= 0:
                sig_bits = 1
            # 控制位 + 前导零数(6位) + 有效位数(6位) + 有效位
            bits.append(1)
            for i in range(5, -1, -1):
                bits.append((leading >> i) & 1)
            for i in range(5, -1, -1):
                bits.append((sig_bits >> i) & 1)
            for i in range(sig_bits - 1, -1, -1):
                bits.append((xor >> (trailing + i)) & 1)
        prev = curr
    # 打包为字节
    result = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            if i + j < len(bits):
                byte |= (bits[i + j] << (7 - j))
        result.append(byte)
    return struct.pack('I', len(values)) + bytes(result)


def gorilla_decode(data: bytes) -> list:
    """解码 Gorilla 压缩数据。"""
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

    first_raw = read_bits(64)
    values = [struct.unpack('d', struct.pack('Q', first_raw))[0]]
    prev = first_raw
    for _ in range(1, count):
        if pos >= len(bits):
            break
        flag = read_bits(1)
        if flag == 0:
            values.append(
                struct.unpack('d', struct.pack('Q', prev))[0])
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
            values.append(
                struct.unpack('d', struct.pack('Q', curr))[0])
            prev = curr
    return values


def gorilla_compression_ratio(values: list) -> float:
    """估算压缩率。"""
    if not values:
        return 1.0
    encoded = gorilla_encode(values)
    original = len(values) * 8
    return len(encoded) / original if original > 0 else 1.0
