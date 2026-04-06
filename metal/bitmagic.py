from __future__ import annotations
"""位级黑魔法 — NaN-Boxing（64位整数支持）、PDEP/PEXT、Broadword Select。"""
import struct
import warnings
# ═══ NaN-Boxing ═══
# 策略变更：INT 不再截断为 32 位。
# 小整数 (|v| < 2^31): INT_TAG | (v & 0xFFFFFFFF)  — 兼容旧路径
# 大整数: 转为 float64 存储（损失精度仅在 |v| > 2^53 时发生）
# 这比截断高位要安全得多。

NULL_TAG  = 0x7FF8000000000001
INT_TAG   = 0x7FF9000000000000
BOOL_TAG  = 0x7FFA000000000000
PTR_TAG   = 0x7FFB000000000000
# [FIX-B03] 新增大整数标签 — 用 float64 编码
BIGINT_FLOAT_TAG = 0  # 标记：直接用 float 编码路径

TAG_MASK   = 0xFFFF000000000000
VALUE_MASK = 0x00000000FFFFFFFF


def nan_pack_float(v: float) -> int:
    """float64 → 64 位无符号整数。"""
    return struct.unpack('Q', struct.pack('d', v))[0]


def nan_unpack_float(bits: int) -> float:
    """64 位无符号整数 → float64。"""
    return struct.unpack('d', struct.pack('Q', bits))[0]


def nan_pack_int(v: int) -> int:
    """整数打包：小整数用 INT_TAG，大整数用 float64 编码。
    |v| > 2^53 时精度可能丢失并发出告警。"""
    # 小整数快速路径（32位范围内）
    if -2147483648 <= v <= 2147483647:
        return INT_TAG | (v & 0xFFFFFFFF)
    # 大整数：转为 float64 存储
    if abs(v) > (1 << 53):
        warnings.warn(
            f"NaN-Boxing: 整数 {v} 超过 2^53，"
            f"转为 float64 可能丢失精度",
            RuntimeWarning, stacklevel=2)
    return nan_pack_float(float(v))


def nan_pack_bool(v: bool) -> int:
    return BOOL_TAG | (1 if v else 0)


def nan_pack_null() -> int:
    return NULL_TAG


def nan_pack_ptr(offset: int) -> int:
    return PTR_TAG | (offset & 0xFFFFFFFF)


def nan_get_tag(bits: int) -> int:
    return bits & TAG_MASK


def nan_is_float(bits: int) -> bool:
    """检查是否为正常 float（非 NaN-boxed 特殊值）。"""
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
    if bits == NULL_TAG:
        return ('NULL', None)
    tag = bits & TAG_MASK
    if tag == INT_TAG:
        v = bits & VALUE_MASK
        if v & 0x80000000:
            v -= 0x100000000
        return ('INT', v)
    if tag == BOOL_TAG:
        return ('BOOL', bool(bits & 1))
    if tag == PTR_TAG:
        return ('PTR', bits & VALUE_MASK)
    # 正常 float — 不做 int 转换（由调用方根据 dtype 决定）
    return ('FLOAT', nan_unpack_float(bits))


# ═══ PDEP / PEXT (Intel BMI2 纯 Python 模拟) ═══

def pdep(source: int, mask: int) -> int:
    """并行位存放。将 source 的低 popcount(mask) 位分散到 mask 的置位位置。"""
    result = 0
    k = 0
    m = mask
    while m:
        lowest = m & (-m)
        if source & (1 << k):
            result |= lowest
        m &= m - 1
        k += 1
    return result


def pext(source: int, mask: int) -> int:
    """并行位提取。将 source 中 mask 置位位置的值压缩到低位。"""
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
    """用位图从列表中提取选中元素（最多 64 个）。"""
    result = []
    for i in range(min(64, len(data_words))):
        if selection_bitmap & (1 << i):
            result.append(data_words[i])
    return result


# ═══ Broadword Select ═══

def select64(word: int, k: int) -> int:
    """找第 k 个（0-indexed）置位位在 64 位字中的位置。
    位数不足返回 -1。"""
    if word == 0:
        return -1
    # SWAR popcount
    s = word
    s = s - ((s >> 1) & 0x5555555555555555)
    s = (s & 0x3333333333333333) + ((s >> 2) & 0x3333333333333333)
    s = (s + (s >> 4)) & 0x0F0F0F0F0F0F0F0F
    prefix = s * 0x0101010101010101
    total = (prefix >> 56) & 0xFF
    if k >= total:
        return -1
    target = k + 1
    byte_idx = 0
    for bi in range(8):
        cum = (prefix >> (bi * 8)) & 0xFF
        if cum >= target:
            byte_idx = bi
            break
    if byte_idx > 0:
        prev_cum = (prefix >> ((byte_idx - 1) * 8)) & 0xFF
        k -= prev_cum
    target_byte = (word >> (byte_idx * 8)) & 0xFF
    pos = 0
    for bit_pos in range(8):
        if target_byte & (1 << bit_pos):
            if k == 0:
                pos = bit_pos
                break
            k -= 1
    return byte_idx * 8 + pos


def rank64(word: int, pos: int) -> int:
    """计算 [0, pos) 范围内的置位数。"""
    if pos <= 0:
        return 0
    if pos >= 64:
        return bin(word & 0xFFFFFFFFFFFFFFFF).count('1')
    mask = (1 << pos) - 1
    return bin(word & mask).count('1')


# ═══ NaN-Boxing 批量操作 ═══

def nanbox_batch_eq(packed_a, packed_b, n: int) -> bytearray:
    """批量等值比较。NULL == NULL → NULL（不置位）。"""
    bmp = bytearray((n + 7) // 8)
    for i in range(n):
        a, b = packed_a[i], packed_b[i]
        if a == NULL_TAG or b == NULL_TAG:
            continue
        if a == b:
            bmp[i >> 3] |= (1 << (i & 7))
    return bmp


def nanbox_batch_lt(packed_a, packed_b, n: int) -> bytearray:
    """批量小于比较。"""
    bmp = bytearray((n + 7) // 8)
    for i in range(n):
        a, b = packed_a[i], packed_b[i]
        if a == NULL_TAG or b == NULL_TAG:
            continue
        _, va = nan_unpack(a)
        _, vb = nan_unpack(b)
        if va is not None and vb is not None and va < vb:
            bmp[i >> 3] |= (1 << (i & 7))
    return bmp


def nanbox_batch_gt(packed_a, packed_b, n: int) -> bytearray:
    """批量大于比较。"""
    bmp = bytearray((n + 7) // 8)
    for i in range(n):
        a, b = packed_a[i], packed_b[i]
        if a == NULL_TAG or b == NULL_TAG:
            continue
        _, va = nan_unpack(a)
        _, vb = nan_unpack(b)
        if va is not None and vb is not None and va > vb:
            bmp[i >> 3] |= (1 << (i & 7))
    return bmp


def nanbox_batch_add(packed_a, packed_b, n: int,
                     is_float: bool = False):
    """批量加法。返回 (result_packed, null_bmp)。"""
    import array
    result = array.array('Q', [0] * n)
    null_bmp = bytearray((n + 7) // 8)
    for i in range(n):
        a, b = packed_a[i], packed_b[i]
        if a == NULL_TAG or b == NULL_TAG:
            result[i] = NULL_TAG
            null_bmp[i >> 3] |= (1 << (i & 7))
            continue
        _, va = nan_unpack(a)
        _, vb = nan_unpack(b)
        if va is not None and vb is not None:
            val = va + vb
            result[i] = (nan_pack_float(float(val))
                         if is_float
                         else nan_pack_int(int(val)))
        else:
            result[i] = NULL_TAG
            null_bmp[i >> 3] |= (1 << (i & 7))
    return result, null_bmp


# ═══ 批量算术 ═══

def nanbox_batch_sub(packed_a, packed_b, n: int,
                     is_float: bool = False):
    """批量减法。返回 (result_packed, null_bmp)。"""
    import array
    result = array.array('Q', [0] * n)
    null_bmp = bytearray((n + 7) // 8)
    for i in range(n):
        a, b = packed_a[i], packed_b[i]
        if a == NULL_TAG or b == NULL_TAG:
            result[i] = NULL_TAG
            null_bmp[i >> 3] |= (1 << (i & 7))
            continue
        _, va = nan_unpack(a)
        _, vb = nan_unpack(b)
        if va is not None and vb is not None:
            val = va - vb
            result[i] = (nan_pack_float(float(val))
                         if is_float else nan_pack_int(int(val)))
        else:
            result[i] = NULL_TAG
            null_bmp[i >> 3] |= (1 << (i & 7))
    return result, null_bmp


def nanbox_batch_mul(packed_a, packed_b, n: int,
                     is_float: bool = False):
    """批量乘法。INT 溢出返回 NULL（由上层报错）。"""
    import array
    result = array.array('Q', [0] * n)
    null_bmp = bytearray((n + 7) // 8)
    for i in range(n):
        a, b = packed_a[i], packed_b[i]
        if a == NULL_TAG or b == NULL_TAG:
            result[i] = NULL_TAG
            null_bmp[i >> 3] |= (1 << (i & 7))
            continue
        _, va = nan_unpack(a)
        _, vb = nan_unpack(b)
        if va is not None and vb is not None:
            val = va * vb
            if is_float:
                result[i] = nan_pack_float(float(val))
            else:
                iv = int(val)
                if -2147483648 <= iv <= 2147483647:
                    result[i] = nan_pack_int(iv)
                else:
                    # 溢出：回退到 float 编码（由 DataVector.get 还原为 int）
                    result[i] = nan_pack_float(float(iv))
        else:
            result[i] = NULL_TAG
            null_bmp[i >> 3] |= (1 << (i & 7))
    return result, null_bmp


def nanbox_batch_pack_int(values: list, null_indices: set,
                          n: int):
    """批量整数打包。返回 packed array.array('Q')。"""
    import array
    packed = array.array('Q', [0] * n)
    for i in range(n):
        if i in null_indices:
            packed[i] = NULL_TAG
        else:
            packed[i] = nan_pack_int(values[i])
    return packed


def nanbox_batch_pack_float(values: list, null_indices: set,
                            n: int):
    """批量浮点打包。返回 packed array.array('Q')。"""
    import array
    packed = array.array('Q', [0] * n)
    for i in range(n):
        if i in null_indices:
            packed[i] = NULL_TAG
        else:
            packed[i] = nan_pack_float(float(values[i]))
    return packed