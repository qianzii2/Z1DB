from __future__ import annotations
"""游程编码 (RLE) — 最适合排序后的低基数列。
相邻重复值合并为 (值, 计数) 对。排序数据压缩率可达 10-1000x。"""
from typing import Any, List, Tuple


def rle_encode(values: list) -> Tuple[list, list]:
    """编码为 (run_values, run_lengths)。
    示例: [1,1,1,2,2,3] → ([1,2,3], [3,2,1])"""
    if not values:
        return [], []
    run_vals: list = [values[0]]
    run_lens: list = [1]
    for i in range(1, len(values)):
        if values[i] == run_vals[-1]:
            run_lens[-1] += 1
        else:
            run_vals.append(values[i])
            run_lens.append(1)
    return run_vals, run_lens


def rle_decode(run_vals: list, run_lens: list) -> list:
    """解码回原始列表。"""
    result: list = []
    for v, n in zip(run_vals, run_lens):
        result.extend([v] * n)
    return result


def rle_aggregate_sum(run_vals: list,
                      run_lens: list) -> Any:
    """直接在 RLE 上求和，无需解压。O(runs) 而非 O(n)。"""
    total = 0
    for v, n in zip(run_vals, run_lens):
        if v is not None:
            total += v * n
    return total


def rle_aggregate_count(run_lens: list) -> int:
    return sum(run_lens)


def rle_filter_eq(run_vals: list, run_lens: list,
                  target: Any) -> list:
    """等值过滤，返回匹配的行索引列表。无需解压。"""
    indices: list = []
    offset = 0
    for v, n in zip(run_vals, run_lens):
        if v == target:
            indices.extend(range(offset, offset + n))
        offset += n
    return indices


def rle_compression_ratio(values: list) -> float:
    """估算压缩率（越低越好）。"""
    if not values:
        return 1.0
    _, run_lens = rle_encode(values)
    return len(run_lens) * 2 / len(values)
