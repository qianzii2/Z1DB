from __future__ import annotations
"""压缩分析器 — 采样数据选择最优编解码器。
对 chunk 中的非 NULL 值采样分析，返回推荐的压缩算法名。"""
from typing import Any, List, Optional
from storage.types import DataType

try:
    from storage.compression.alp import alp_compression_ratio
    _HAS_ALP = True
except ImportError:
    _HAS_ALP = False


def analyze_and_choose(values: list, dtype: DataType,
                       sample_size: int = 1024) -> str:
    """采样并选择最佳压缩方法。
    返回：'NONE','RLE','DICT','DELTA','FOR','GORILLA','ALP'。"""
    if not values:
        return 'NONE'
    sample = values[:sample_size]
    non_null = [v for v in sample if v is not None]
    if not non_null:
        return 'NONE'

    # 字符串列
    if dtype in (DataType.VARCHAR, DataType.TEXT):
        ndv = len(set(non_null))
        if ndv < min(len(non_null), 65536):
            return 'DICT'
        return 'NONE'

    # 布尔列：不压缩（Bitmap 已足够紧凑）
    if dtype == DataType.BOOLEAN:
        return 'NONE'

    # 浮点列
    if dtype in (DataType.FLOAT, DataType.DOUBLE):
        # 优先 ALP（科学/金融数据压缩率更好）
        if _HAS_ALP and len(non_null) > 10:
            try:
                alp_ratio = alp_compression_ratio(
                    [float(v) for v in non_null])
                if alp_ratio < 0.4:
                    return 'ALP'
            except Exception:
                pass
        # 回退 Gorilla
        ratio = _try_gorilla(non_null)
        if ratio < 0.5:
            return 'GORILLA'
        return 'NONE'

    # 整数列
    if dtype in (DataType.INT, DataType.BIGINT,
                 DataType.DATE, DataType.TIMESTAMP):
        # 已排序 → 差分编码
        sorted_check = all(
            non_null[i] <= non_null[i + 1]
            for i in range(len(non_null) - 1))
        if sorted_check:
            return 'DELTA'
        # 大量重复 → RLE
        rle_ratio = _try_rle(non_null)
        if rle_ratio < 0.3:
            return 'RLE'
        # 值域窄 → FOR
        for_ratio = _try_for(non_null)
        if for_ratio < 0.5:
            return 'FOR'
        # 低基数 → DICT
        ndv = len(set(non_null))
        if ndv < min(len(non_null) // 2, 65536):
            return 'DICT'
        return 'NONE'

    return 'NONE'


def _try_rle(values: list) -> float:
    if not values:
        return 1.0
    runs = 1
    for i in range(1, len(values)):
        if values[i] != values[i - 1]:
            runs += 1
    return runs / len(values)


def _try_for(values: list) -> float:
    if not values:
        return 1.0
    min_v = min(values)
    max_v = max(values)
    range_v = max_v - min_v
    if range_v == 0:
        return 1.0 / 64
    bit_width = max(1, range_v.bit_length())
    return bit_width / 64


def _try_gorilla(values: list) -> float:
    if len(values) < 2:
        return 1.0
    try:
        from storage.compression.gorilla import gorilla_compression_ratio
        return gorilla_compression_ratio(values)
    except Exception:
        return 1.0
