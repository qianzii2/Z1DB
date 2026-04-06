from __future__ import annotations
"""压缩执行 — 直接在压缩/字典数据上操作。
[修复] AVG 跨 chunk 用 (sum, count) 元组精确合并。"""
from typing import Any, Dict, List, Optional, Tuple
from metal.bitmap import Bitmap


# ═══ RLE 直接聚合 ═══

def rle_aggregate_direct(run_vals: list, run_lens: list,
                         agg_type: str) -> Any:
    """直接在 RLE 编码数据上聚合，O(runs)。"""
    if not run_vals:
        return None
    upper = agg_type.upper()
    if upper == 'COUNT':
        return sum(run_lens)
    if upper == 'SUM':
        total = 0
        for v, l in zip(run_vals, run_lens):
            if v is not None:
                total += v * l
        return total
    if upper == 'MIN':
        vals = [v for v in run_vals if v is not None]
        return min(vals) if vals else None
    if upper == 'MAX':
        vals = [v for v in run_vals if v is not None]
        return max(vals) if vals else None
    if upper == 'AVG':
        total = count = 0
        for v, l in zip(run_vals, run_lens):
            if v is not None:
                total += v * l
                count += l
        return total / count if count > 0 else None
    return None


def rle_filter_eq_direct(run_vals: list, run_lens: list,
                         target: Any) -> Bitmap:
    """RLE 等值过滤，返回匹配位图。"""
    total = sum(run_lens)
    bm = Bitmap(total)
    offset = 0
    for v, l in zip(run_vals, run_lens):
        if v == target:
            for i in range(offset, offset + l):
                bm.set_bit(i)
        offset += l
    return bm


# ═══ 字典编码直接操作 ═══

def dict_filter_eq(codes: Any, target_code: int, n: int) -> Bitmap:
    """字典编码列等值过滤。"""
    bm = Bitmap(n)
    if hasattr(codes, '__getitem__'):
        for i in range(n):
            if codes[i] == target_code:
                bm.set_bit(i)
    return bm


def dict_group_aggregate(codes: Any, values: Any, n: int,
                         ndv: int, agg_type: str) -> Dict[int, Any]:
    """字典编码 GROUP BY 聚合。
    AVG 返回 (sum, count) 元组，由调用方合并。
    其他聚合返回最终值。"""
    upper = agg_type.upper()

    if upper == 'COUNT':
        counts = [0] * ndv
        for i in range(n):
            counts[codes[i]] += 1
        return {c: cnt for c, cnt in enumerate(counts) if cnt > 0}

    if upper == 'SUM':
        sums = [0] * ndv
        has = [False] * ndv
        for i in range(n):
            c = codes[i]
            if values is not None and i < len(values) and values[i] is not None:
                sums[c] += values[i]
                has[c] = True
        return {c: s for c, s in enumerate(sums) if has[c]}

    if upper == 'AVG':
        # [修复] 返回 (sum, count) 元组，支持跨 chunk 精确合并
        sums = [0.0] * ndv
        counts = [0] * ndv
        for i in range(n):
            c = codes[i]
            if values is not None and i < len(values) and values[i] is not None:
                sums[c] += values[i]
                counts[c] += 1
        return {c: (sums[c], counts[c])
                for c in range(ndv) if counts[c] > 0}

    if upper == 'MIN':
        mins: list = [None] * ndv
        for i in range(n):
            c = codes[i]
            v = values[i] if values is not None and i < len(values) else None
            if v is not None and (mins[c] is None or v < mins[c]):
                mins[c] = v
        return {c: m for c, m in enumerate(mins) if m is not None}

    if upper == 'MAX':
        maxs: list = [None] * ndv
        for i in range(n):
            c = codes[i]
            v = values[i] if values is not None and i < len(values) else None
            if v is not None and (maxs[c] is None or v > maxs[c]):
                maxs[c] = v
        return {c: m for c, m in enumerate(maxs) if m is not None}

    return {}


# ═══ 检测函数 ═══

def can_use_dict_execution(vec: Any) -> bool:
    return hasattr(vec, 'dict_encoded') and vec.dict_encoded is not None


def can_use_rle_execution(vec: Any) -> bool:
    return hasattr(vec, '_rle_encoded') and vec._rle_encoded is not None


# ═══ 跨 chunk 聚合 ═══

def try_rle_aggregate(store: Any, col_name: str,
                      agg_type: str) -> Optional[Any]:
    """跨所有 chunk 在 RLE 上聚合。失败返回 None。"""
    if not hasattr(store, 'get_column_chunks'):
        return None
    try:
        chunks = store.get_column_chunks(col_name)
        if not chunks:
            return None
        all_rv = []
        all_rl = []
        for chunk in chunks:
            if chunk.row_count == 0:
                continue
            rle = chunk.get_rle_data()
            if rle is None:
                return None  # 任一 chunk 无 RLE → 放弃
            rv, rl = rle
            all_rv.extend(rv)
            all_rl.extend(rl)
        if not all_rv:
            return None
        return rle_aggregate_direct(all_rv, all_rl, agg_type)
    except Exception:
        return None


def try_dict_group_aggregate(store: Any, key_col: str,
                             val_col: Optional[str],
                             agg_type: str) -> Optional[Dict[Any, Any]]:
    """跨所有 chunk 做字典编码 GROUP BY。
    [修复] AVG 用 (sum, count) 精确合并。
    成功返回 {key_value: agg_result}，失败返回 None。"""
    if not hasattr(store, 'get_column_chunks'):
        return None
    try:
        key_chunks = store.get_column_chunks(key_col)
        val_chunks = (store.get_column_chunks(val_col)
                      if val_col else None)
        upper = agg_type.upper()

        # 收集每个 chunk 的部分结果
        # key_value → list of partial results
        partials: Dict[Any, list] = {}

        for ci, kc in enumerate(key_chunks):
            if kc.row_count == 0:
                continue
            if kc.dict_encoded is None:
                return None  # 任一 chunk 无字典 → 放弃
            de = kc.dict_encoded
            n = kc.row_count

            val_list = None
            if val_chunks and ci < len(val_chunks):
                vc = val_chunks[ci]
                val_list = [vc.get(i) for i in range(vc.row_count)]

            code_results = dict_group_aggregate(
                de.codes, val_list, n, de.ndv, agg_type)

            for code, agg_val in code_results.items():
                if code < len(de.dictionary):
                    key_val = de.dictionary[code]
                    if key_val not in partials:
                        partials[key_val] = []
                    partials[key_val].append(agg_val)

        if not partials:
            return None

        # 合并跨 chunk 的部分结果
        final: Dict[Any, Any] = {}
        for key_val, parts in partials.items():
            if upper == 'COUNT':
                final[key_val] = sum(parts)
            elif upper == 'SUM':
                final[key_val] = sum(parts)
            elif upper == 'MIN':
                final[key_val] = min(parts)
            elif upper == 'MAX':
                final[key_val] = max(parts)
            elif upper == 'AVG':
                # [修复] parts 中每个元素是 (sum, count) 元组
                total_sum = sum(p[0] for p in parts)
                total_count = sum(p[1] for p in parts)
                final[key_val] = (total_sum / total_count
                                  if total_count > 0 else None)
            else:
                final[key_val] = parts[0]
        return final
    except Exception:
        return None
