from __future__ import annotations

"""SWAR (SIMD Within A Register) — process 8 bytes in one Python int op."""

_LO = 0x0101010101010101
_HI = 0x8080808080808080
_7F = 0x7F7F7F7F7F7F7F7F
_20 = 0x2020202020202020


def pack_8_bytes(b0: int, b1: int, b2: int, b3: int,
                 b4: int, b5: int, b6: int, b7: int) -> int:
    return (b0 | (b1 << 8) | (b2 << 16) | (b3 << 24) |
            (b4 << 32) | (b5 << 40) | (b6 << 48) | (b7 << 56))


def pack_bytes(data: bytes) -> int:
    result = 0
    for i, b in enumerate(data[:8]):
        result |= (b << (i * 8))
    return result


def unpack_bytes(word: int, count: int = 8) -> bytes:
    return bytes((word >> (i * 8)) & 0xFF for i in range(count))


def has_zero_byte(x: int) -> bool:
    """O(1) detect if any of 8 bytes is zero. Mycroft, 1987."""
    return bool(((x - _LO) & ~x & _HI))


def has_byte_equal_to(x: int, n: int) -> bool:
    return has_zero_byte(x ^ (n * _LO))


def find_byte(x: int, n: int) -> int:
    """Position of first byte equal to n. Returns -1 if not found."""
    diff = x ^ (n * _LO)
    mask = (diff - _LO) & ~diff & _HI
    if mask == 0:
        return -1
    return _ctz_byte(mask)


def to_upper_ascii_8(x: int) -> int:
    """Convert 8 ASCII bytes to uppercase simultaneously.

    Correct SWAR range detection:
    1. Mask bit 7 to prevent inter-byte carry (max byte = 0x7F)
    2. Add offset to set bit 7 if byte >= 'a' (0x61): offset = 0x80 - 0x61 = 0x1F
    3. Add offset to set bit 7 if byte > 'z' (0x7A): offset = 0x80 - 0x7B = 0x05
    4. is_lower = ge_a AND NOT gt_z
    5. Shift bit 7 → bit 5 → XOR to flip case

    Max value: 0x7F + 0x3F = 0xBE < 0x100 → no inter-byte carry.
    """
    x_lo = x & _7F
    ge_a = (x_lo + 0x1F1F1F1F1F1F1F1F) & _HI  # bit 7 set if byte >= 0x61
    gt_z = (x_lo + 0x0505050505050505) & _HI  # bit 7 set if byte > 0x7A
    is_lower = ge_a & ~gt_z
    mask = is_lower >> 2  # bit 7 → bit 5 = 0x20
    return x ^ mask


def to_lower_ascii_8(x: int) -> int:
    """Convert 8 ASCII bytes to lowercase simultaneously.
    Same technique: detect 'A'(0x41)-'Z'(0x5A) range, flip bit 5."""
    x_lo = x & _7F
    ge_A = (x_lo + 0x3F3F3F3F3F3F3F3F) & _HI  # bit 7 set if byte >= 0x41
    gt_Z = (x_lo + 0x2525252525252525) & _HI  # bit 7 set if byte > 0x5A
    is_upper = ge_A & ~gt_Z
    mask = is_upper >> 2
    return x ^ mask


def count_spaces_8(x: int) -> int:
    diff = x ^ (0x20 * _LO)
    zero_mask = (diff - _LO) & ~diff & _HI
    return bin(zero_mask).count('1')


def batch_to_upper(data: bytearray) -> bytearray:
    """SWAR-accelerated UPPER() for byte strings."""
    result = bytearray(len(data))
    i = 0
    n = len(data)
    while i + 8 <= n:
        word = int.from_bytes(data[i:i + 8], 'little')
        upper_word = to_upper_ascii_8(word)
        result[i:i + 8] = upper_word.to_bytes(8, 'little')
        i += 8
    while i < n:
        b = data[i]
        if 0x61 <= b <= 0x7A:
            result[i] = b - 0x20
        else:
            result[i] = b
        i += 1
    return result


def batch_to_lower(data: bytearray) -> bytearray:
    """SWAR-accelerated LOWER() for byte strings."""
    result = bytearray(len(data))
    i = 0
    n = len(data)
    while i + 8 <= n:
        word = int.from_bytes(data[i:i + 8], 'little')
        lower_word = to_lower_ascii_8(word)
        result[i:i + 8] = lower_word.to_bytes(8, 'little')
        i += 8
    while i < n:
        b = data[i]
        if 0x41 <= b <= 0x5A:
            result[i] = b + 0x20
        else:
            result[i] = b
        i += 1
    return result


def batch_find_char(data: bytes, char: int) -> list[int]:
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
    """Position of first set high-bit (byte index)."""
    if mask == 0:
        return 8
    n = 0
    if (mask & 0x00000000FFFFFFFF) == 0:
        n += 4;
        mask >>= 32
    if (mask & 0x0000FFFF) == 0:
        n += 2;
        mask >>= 16
    if (mask & 0x00FF) == 0:
        n += 1
    return n
