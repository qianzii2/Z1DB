from __future__ import annotations
"""SWAR（SIMD Within A Register）— 一个 Python int 操作同时处理 8 字节。
原理：将 8 个 ASCII 字节打包到一个 64 位 int 中，用位运算并行处理。
注意：SWAR 在 CPython 中仅对 >100 字节的纯 ASCII 数据有显著收益。
短字符串（<16 字节）应直接用 str.upper()/str.lower()。"""

_LO = 0x0101010101010101
_HI = 0x8080808080808080
_7F = 0x7F7F7F7F7F7F7F7F
_20 = 0x2020202020202020

# 短字符串阈值：低于此长度直接用 Python 字符串方法 [P06]
_SHORT_THRESHOLD = 16


def pack_8_bytes(b0: int, b1: int, b2: int, b3: int,
                 b4: int, b5: int, b6: int, b7: int) -> int:
    """将 8 个字节打包为一个 64 位整数（小端序）。"""
    return (b0 | (b1 << 8) | (b2 << 16) | (b3 << 24) |
            (b4 << 32) | (b5 << 40) | (b6 << 48) | (b7 << 56))


def pack_bytes(data: bytes) -> int:
    """将字节序列打包为 64 位整数（最多取 8 字节）。"""
    result = 0
    for i, b in enumerate(data[:8]):
        result |= (b << (i * 8))
    return result


def unpack_bytes(word: int, count: int = 8) -> bytes:
    """将 64 位整数拆包为字节序列。"""
    return bytes((word >> (i * 8)) & 0xFF
                 for i in range(count))


def has_zero_byte(x: int) -> bool:
    """O(1) 检测 8 字节中是否有零字节。Mycroft, 1987。"""
    return bool(((x - _LO) & ~x & _HI))


def has_byte_equal_to(x: int, n: int) -> bool:
    """O(1) 检测 8 字节中是否有等于 n 的字节。"""
    return has_zero_byte(x ^ (n * _LO))


def find_byte(x: int, n: int) -> int:
    """找第一个等于 n 的字节位置。未找到返回 -1。"""
    diff = x ^ (n * _LO)
    mask = (diff - _LO) & ~diff & _HI
    if mask == 0:
        return -1
    return _ctz_byte(mask)


def to_upper_ascii_8(x: int) -> int:
    """并行将 8 个 ASCII 字节转为大写。
    步骤：
      1. 屏蔽 bit7 防止进位跨字节
      2. 加偏移量检测 >= 'a'(0x61)：offset = 0x80 - 0x61 = 0x1F
      3. 加偏移量检测 > 'z'(0x7A)：offset = 0x80 - 0x7B = 0x05
      4. is_lower = ge_a AND NOT gt_z
      5. bit7 → bit5 → XOR 翻转大小写
    最大值：0x7F + 0x3F = 0xBE < 0x100 → 无跨字节进位。"""
    x_lo = x & _7F
    ge_a = (x_lo + 0x1F1F1F1F1F1F1F1F) & _HI
    gt_z = (x_lo + 0x0505050505050505) & _HI
    is_lower = ge_a & ~gt_z
    mask = is_lower >> 2  # bit7 → bit5 = 0x20
    return x ^ mask


def to_lower_ascii_8(x: int) -> int:
    """并行将 8 个 ASCII 字节转为小写。
    检测 'A'(0x41)-'Z'(0x5A) 范围，翻转 bit5。"""
    x_lo = x & _7F
    ge_A = (x_lo + 0x3F3F3F3F3F3F3F3F) & _HI
    gt_Z = (x_lo + 0x2525252525252525) & _HI
    is_upper = ge_A & ~gt_Z
    mask = is_upper >> 2
    return x ^ mask


def count_spaces_8(x: int) -> int:
    """O(1) 统计 8 字节中空格(0x20)的数量。"""
    diff = x ^ (0x20 * _LO)
    zero_mask = (diff - _LO) & ~diff & _HI
    return bin(zero_mask).count('1')


def batch_to_upper(data: bytearray) -> bytearray:
    """SWAR 加速的 UPPER()。短数据直接用 Python 方法 [P06]。"""
    n = len(data)
    if n < _SHORT_THRESHOLD:
        # 短数据：Python 逐字节转换更快（避免 int↔bytes 转换开销）
        result = bytearray(n)
        for i in range(n):
            b = data[i]
            result[i] = b - 0x20 if 0x61 <= b <= 0x7A else b
        return result
    result = bytearray(n)
    i = 0
    while i + 8 <= n:
        word = int.from_bytes(data[i:i + 8], 'little')
        upper_word = to_upper_ascii_8(word)
        result[i:i + 8] = upper_word.to_bytes(8, 'little')
        i += 8
    while i < n:
        b = data[i]
        result[i] = b - 0x20 if 0x61 <= b <= 0x7A else b
        i += 1
    return result


def batch_to_lower(data: bytearray) -> bytearray:
    """SWAR 加速的 LOWER()。短数据直接用 Python 方法 [P06]。"""
    n = len(data)
    if n < _SHORT_THRESHOLD:
        result = bytearray(n)
        for i in range(n):
            b = data[i]
            result[i] = b
            result[i] = b + 0x20 if 0x41 <= b <= 0x5A else b
        return result
    result = bytearray(n)
    i = 0
    while i + 8 <= n:
        word = int.from_bytes(data[i:i + 8], 'little')
        lower_word = to_lower_ascii_8(word)
        result[i:i + 8] = lower_word.to_bytes(8, 'little')
        i += 8
    while i < n:
        b = data[i]
        result[i] = b + 0x20 if 0x41 <= b <= 0x5A else b
        i += 1
    return result


def batch_find_char(data: bytes, char: int) -> list[int]:
    """在字节序列中查找所有指定字符的位置。"""
    positions: list[int] = []
    i = 0
    n = len(data)
    broadcast = char * _LO
    while i + 8 <= n:
        word = int.from_bytes(data[i:i + 8], 'little')
        diff = word ^ broadcast
        mask = (diff - _LO) & ~diff & _HI
        while mask:
            pos = _ctz_byte(mask)
            positions.append(i + pos)
            mask &= mask - (1 << (pos * 8 + 7))
        i += 8
    while i < n:
        if data[i] == char:
            positions.append(i)
        i += 1
    return positions


def _ctz_byte(mask: int) -> int:
    """找第一个高位被设置的字节位置（即第一个匹配字节的索引）。"""
    if mask == 0:
        return 8
    n = 0
    if (mask & 0x00000000FFFFFFFF) == 0:
        n += 4; mask >>= 32
    if (mask & 0x0000FFFF) == 0:
        n += 2; mask >>= 16
    if (mask & 0x00FF) == 0:
        n += 1
    return n
