from __future__ import annotations

"""Bit-level black magic — NaN-Boxing, PDEP/PEXT, Broadword Select.

NaN-Boxing: pack type+value into 64-bit IEEE 754 NaN space.
PDEP/PEXT: parallel bit deposit/extract (Intel BMI2 in pure Python).
Broadword Select: O(1) find k-th set bit. Paper: Vigna, 2008.
"""
import struct

# ═══ NaN-Boxing ═══
# IEEE 754 double has 2^52 NaN encodings — we steal them for tagged values.
#
# Normal float:    unchanged
# NULL:            0x7FF8_0000_0000_0001
# INT32(v):        0x7FF9_0000_VVVV_VVVV
# BOOL(v):         0x7FFA_0000_0000_000V
# STRING_PTR(p):   0x7FFB_0000_PPPP_PPPP

NULL_TAG = 0x7FF8000000000001
INT_TAG = 0x7FF9000000000000
BOOL_TAG = 0x7FFA000000000000
PTR_TAG = 0x7FFB000000000000
TAG_MASK = 0xFFFF000000000000
VALUE_MASK = 0x00000000FFFFFFFF


def nan_pack_float(v: float) -> int:
    """Pack float64 → 64-bit int. Normal float path."""
    return struct.unpack('Q', struct.pack('d', v))[0]


def nan_unpack_float(bits: int) -> float:
    """Unpack 64-bit int → float64."""
    return struct.unpack('d', struct.pack('Q', bits))[0]


def nan_pack_int(v: int) -> int:
    """Pack int32 into NaN space."""
    return INT_TAG | (v & 0xFFFFFFFF)


def nan_pack_bool(v: bool) -> int:
    return BOOL_TAG | (1 if v else 0)


def nan_pack_null() -> int:
    return NULL_TAG


def nan_pack_ptr(offset: int) -> int:
    return PTR_TAG | (offset & 0xFFFFFFFF)


def nan_get_tag(bits: int) -> int:
    return bits & TAG_MASK


def nan_is_float(bits: int) -> bool:
    """Check if the value is a normal (non-NaN-boxed) float."""
    # Not a NaN → it's a float; or it's ±0, ±inf
    tag = (bits >> 48) & 0xFFFF
    return tag < 0x7FF8 or tag > 0x7FFF


def nan_is_null(bits: int) -> bool:
    return bits == NULL_TAG


def nan_is_int(bits: int) -> bool:
    return (bits & TAG_MASK) == INT_TAG


def nan_is_bool(bits: int) -> bool:
    return (bits & TAG_MASK) == BOOL_TAG


def nan_is_ptr(bits: int) -> bool:
    return (bits & TAG_MASK) == PTR_TAG


def nan_unpack(bits: int) -> tuple:
    """Unpack → (type_str, value)."""
    if bits == NULL_TAG:
        return ('NULL', None)
    tag = bits & TAG_MASK
    if tag == INT_TAG:
        v = bits & VALUE_MASK
        # Sign-extend from 32 bits
        if v & 0x80000000:
            v -= 0x100000000
        return ('INT', v)
    if tag == BOOL_TAG:
        return ('BOOL', bool(bits & 1))
    if tag == PTR_TAG:
        return ('PTR', bits & VALUE_MASK)
    # Normal float
    return ('FLOAT', nan_unpack_float(bits))


# ═══ PDEP / PEXT (Intel BMI2 emulation) ═══

def pdep(source: int, mask: int) -> int:
    """Parallel Bit Deposit.
    Scatter the low popcount(mask) bits of source to the positions of 1-bits in mask.

    Example: pdep(0b1011, 0b11010100) = 0b10010100
    """
    result = 0
    k = 0
    m = mask
    while m:
        lowest = m & (-m)  # isolate lowest set bit
        if source & (1 << k):
            result |= lowest
        m &= m - 1  # clear lowest set bit
        k += 1
    return result


def pext(source: int, mask: int) -> int:
    """Parallel Bit Extract.
    Compress the bits of source at positions of 1-bits in mask to contiguous low bits.

    Inverse of pdep.
    Example: pext(0b10010100, 0b11010100) = 0b1011
    """
    result = 0
    k = 0
    m = mask
    while m:
        lowest = m & (-m)
        if source & lowest:
            result |= (1 << k)
        m &= m - 1
        k += 1
    return result


def bitmap_gather(data_words: list, selection_bitmap: int) -> list:
    """Use PEXT to gather selected elements from 64-element groups."""
    result = []
    bit = 0
    for i in range(min(64, len(data_words))):
        if selection_bitmap & (1 << i):
            result.append(data_words[i])
    return result


# ═══ Broadword Select ═══
# O(1) find the position of the k-th set bit in a 64-bit word.
# Paper: Vigna, 2008 "Broadword Implementation of Rank/Select Queries"

def select64(word: int, k: int) -> int:
    """Find position of the k-th (0-indexed) set bit in a 64-bit word.
    Returns -1 if fewer than k+1 bits are set.
    Uses broadword programming for O(1) with small constant."""
    if word == 0:
        return -1

    # Byte-level popcount via SWAR
    s = word
    s = s - ((s >> 1) & 0x5555555555555555)
    s = (s & 0x3333333333333333) + ((s >> 2) & 0x3333333333333333)
    s = (s + (s >> 4)) & 0x0F0F0F0F0F0F0F0F

    # Prefix sum of byte popcounts (multiply by 0x0101... = prefix sum trick)
    prefix = s * 0x0101010101010101

    # Total popcount
    total = (prefix >> 56) & 0xFF
    if k >= total:
        return -1

    # Find which byte contains the k-th bit
    # Compare prefix sums against k+1
    target = k + 1
    byte_idx = 0

    # Check each byte's cumulative popcount
    for bi in range(8):
        cum = (prefix >> (bi * 8)) & 0xFF
        if cum >= target:
            byte_idx = bi
            break

    # Adjust k for bits in previous bytes
    if byte_idx > 0:
        prev_cum = (prefix >> ((byte_idx - 1) * 8)) & 0xFF
        k -= prev_cum

    # Extract target byte
    target_byte = (word >> (byte_idx * 8)) & 0xFF

    # Find k-th set bit within the byte (small lookup)
    pos = 0
    for bit_pos in range(8):
        if target_byte & (1 << bit_pos):
            if k == 0:
                pos = bit_pos
                break
            k -= 1

    return byte_idx * 8 + pos


def rank64(word: int, pos: int) -> int:
    """Count set bits in positions [0, pos). O(1) with popcount."""
    if pos <= 0:
        return 0
    if pos >= 64:
        return bin(word & 0xFFFFFFFFFFFFFFFF).count('1')
    mask = (1 << pos) - 1
    return bin(word & mask).count('1')
