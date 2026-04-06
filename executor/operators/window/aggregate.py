from __future__ import annotations
"""窗口聚合函数 — 6 种优化路径 + 暴力回退。"""
from typing import Any, List, Optional
from executor.operators.window.ranking import frame_bounds

try:
    from structures.fenwick_tree import FenwickTree
    _HAS_FENWICK = True
except ImportError: _HAS_FENWICK = False
try:
    from structures.segment_tree import SegmentTree
    _HAS_SEGTREE = True
except ImportError: _HAS_SEGTREE = False
try:
    from structures.sparse_table import SparseTableMin, SparseTableMax
    _HAS_SPARSE = True
except ImportError: _HAS_SPARSE = False
try:
    from structures.wavelet_tree import WaveletTree
    _HAS_WAVELET = True
except ImportError: _HAS_WAVELET = False
try:
    from structures.sorted_container import SortedList
    _HAS_SORTED = True
except ImportError: _HAS_SORTED = False


def compute_agg_window(fn, frame, indices, batch, results, ho,
                       evaluator):
    """聚合窗口函数入口。自动选择最优计算路径。"""
    from parser.ast import StarExpr
    name = fn.name.upper()
    if fn.args and not isinstance(fn.args[0], StarExpr):
        av = evaluator.evaluate(fn.args[0], batch).to_python_list()
    else:
        av = None
    ps = len(indices)

    vals = []
    has_nulls = False
    for p in range(ps):
        if av is not None:
            v = av[indices[p]]
            if v is None: has_nulls = True
            vals.append(v if v is not None else 0)
        else:
            vals.append(1)

    fst = frame.start.type if frame and frame.start else 'UNBOUNDED_PRECEDING'
    fen = frame.end.type if frame and frame.end else 'CURRENT_ROW'

    # 无 frame + 无 ORDER BY → 整个分区
    if frame is None and not ho:
        _agg_whole(name, vals, indices, results, av, ps); return
    # 前缀累加路径
    if frame is None and ho:
        _agg_prefix(name, vals, indices, results, av, ps); return
    if fst == 'UNBOUNDED_PRECEDING' and fen == 'CURRENT_ROW':
        _agg_prefix(name, vals, indices, results, av, ps); return

    # ★ 新增：滑动窗口路径（SUM/COUNT/AVG，O(N)）
    if _can_use_sliding(name, frame, has_nulls) and ps > 16:
        _agg_sliding(name, vals, indices, results, av, frame, ps, ho)
        return


# ═══ 分区全量 ═══

def _agg_whole(name, vals, indices, results, av, ps):
    if name == 'COUNT':
        c = sum(1 for p in range(ps) if av is None or av[indices[p]] is not None)
        for i in indices: results[i] = c
    elif name == 'SUM':
        s = sum(vals[p] for p in range(ps) if av is None or av[indices[p]] is not None)
        for i in indices: results[i] = s
    elif name == 'AVG':
        t = c = 0
        for p in range(ps):
            if av is None or av[indices[p]] is not None: t += vals[p]; c += 1
        a = t / c if c > 0 else None
        for i in indices: results[i] = a
    elif name == 'MIN':
        nn = [vals[p] for p in range(ps) if av is None or av[indices[p]] is not None]
        m = min(nn) if nn else None
        for i in indices: results[i] = m
    elif name == 'MAX':
        nn = [vals[p] for p in range(ps) if av is None or av[indices[p]] is not None]
        m = max(nn) if nn else None
        for i in indices: results[i] = m


# ═══ 前缀累加 ═══

def _agg_prefix(name, vals, indices, results, av, ps):
    if name == 'SUM' and _HAS_FENWICK and ps > 100:
        has_nulls = av is not None and any(av[indices[p]] is None for p in range(ps))
        if not has_nulls:
            ft = FenwickTree.from_list(vals)
            for p, i in enumerate(indices): results[i] = ft.prefix_sum(p)
            return
    if name == 'COUNT':
        c = 0
        for p, i in enumerate(indices):
            if av is None or av[i] is not None: c += 1
            results[i] = c
    elif name == 'SUM':
        s = 0
        for p, i in enumerate(indices):
            if av is None or av[i] is not None: s += vals[p]
            results[i] = s
    elif name == 'AVG':
        s = c = 0
        for p, i in enumerate(indices):
            if av is None or av[i] is not None: s += vals[p]; c += 1
            results[i] = s / c if c > 0 else None
    elif name == 'MIN':
        cm = None
        for p, i in enumerate(indices):
            if av is None or av[i] is not None:
                if cm is None or vals[p] < cm: cm = vals[p]
            results[i] = cm
    elif name == 'MAX':
        cm = None
        for p, i in enumerate(indices):
            if av is None or av[i] is not None:
                if cm is None or vals[p] > cm: cm = vals[p]
            results[i] = cm


# ═══ FenwickTree ═══

def _agg_fenwick(name, vals, indices, results, frame, ps, ho):
    ft = FenwickTree.from_list(vals)
    for p in range(ps):
        s, e = frame_bounds(frame, p, ps, ho)
        rs = ft.range_sum(s, e)
        if name == 'SUM': results[indices[p]] = rs
        elif name == 'COUNT': results[indices[p]] = e - s + 1
        elif name == 'AVG':
            c = e - s + 1; results[indices[p]] = rs / c if c > 0 else None


# ═══ SegmentTree ═══

def _agg_segtree(name, vals, indices, results, frame, ps, ho):
    st = SegmentTree(vals)
    for p in range(ps):
        s, e = frame_bounds(frame, p, ps, ho)
        rs = st.query(s, e)
        if name == 'SUM': results[indices[p]] = rs
        elif name == 'COUNT': results[indices[p]] = e - s + 1
        elif name == 'AVG':
            c = e - s + 1; results[indices[p]] = rs / c if c > 0 else None


# ═══ SparseTable ═══

def _agg_sparse(name, vals, indices, results, frame, ps, ho):
    st = SparseTableMin(vals) if name == 'MIN' else SparseTableMax(vals)
    for p in range(ps):
        s, e = frame_bounds(frame, p, ps, ho)
        results[indices[p]] = st.query(s, e)


# ═══ WaveletTree ═══

def _agg_wavelet(name, vals, indices, results, frame, ps, ho):
    sorted_unique = sorted(set(vals))
    val_to_int = {v: i for i, v in enumerate(sorted_unique)}
    int_vals = [val_to_int[v] for v in vals]
    sigma = len(sorted_unique)
    if sigma == 0: return
    wt = WaveletTree(int_vals, sigma)
    for p in range(ps):
        s, e = frame_bounds(frame, p, ps, ho)
        count = e - s + 1
        if count <= 0: results[indices[p]] = None; continue
        if name == 'MEDIAN':
            if count % 2 == 1:
                iv = wt.quantile(s, e + 1, count // 2)
                results[indices[p]] = sorted_unique[iv] if 0 <= iv < sigma else None
            else:
                iv1 = wt.quantile(s, e + 1, count // 2 - 1)
                iv2 = wt.quantile(s, e + 1, count // 2)
                if 0 <= iv1 < sigma and 0 <= iv2 < sigma:
                    results[indices[p]] = (sorted_unique[iv1] + sorted_unique[iv2]) / 2
                else: results[indices[p]] = None
        else:
            k = max(0, min(count - 1, count // 2))
            iv = wt.quantile(s, e + 1, k)
            results[indices[p]] = sorted_unique[iv] if 0 <= iv < sigma else None


# ═══ Mo's + SortedContainer ═══

def _agg_mos_median(vals, indices, results, frame, ps, ho):
    sl = SortedList(); cur_s = cur_e = 0
    for p in range(ps):
        s, e = frame_bounds(frame, p, ps, ho)
        while cur_s < s: sl.remove(vals[cur_s]); cur_s += 1
        while cur_s > s: cur_s -= 1; sl.add(vals[cur_s])
        while cur_e <= e: sl.add(vals[cur_e]); cur_e += 1
        while cur_e > e + 1: cur_e -= 1; sl.remove(vals[cur_e])
        results[indices[p]] = sl.median()


# ═══ 暴力回退 ═══

def _agg_brute(name, vals, indices, results, av, frame, ps, ho):
    for p in range(ps):
        s, e = frame_bounds(frame, p, ps, ho)
        wv = []
        for wp in range(s, e + 1):
            if av is not None:
                v = av[indices[wp]]
                if v is not None: wv.append(v)
            else: wv.append(1)
        if name == 'COUNT': results[indices[p]] = len(wv)
        elif name == 'SUM': results[indices[p]] = sum(wv) if wv else None
        elif name == 'AVG': results[indices[p]] = sum(wv) / len(wv) if wv else None
        elif name == 'MIN': results[indices[p]] = min(wv) if wv else None
        elif name == 'MAX': results[indices[p]] = max(wv) if wv else None
        elif name == 'MEDIAN':
            if wv:
                wv.sort(); m = len(wv)
                results[indices[p]] = wv[m // 2] if m % 2 == 1 else (wv[m // 2 - 1] + wv[m // 2]) / 2
            else: results[indices[p]] = None

def _agg_sliding(name, vals, indices, results, av, frame, ps, ho):
    """SUM/COUNT/AVG 的滑动窗口 O(N) 计算。
    当帧是 N PRECEDING 到 M FOLLOWING 形式时使用。"""
    # 计算首个窗口
    s0, e0 = frame_bounds(frame, 0, ps, ho)
    window_sum = 0.0
    window_count = 0
    for wp in range(s0, min(e0 + 1, ps)):
        if av is not None:
            v = av[indices[wp]]
            if v is not None:
                window_sum += v
                window_count += 1
        else:
            window_sum += 1
            window_count += 1

    _emit(name, indices[0], results, window_sum, window_count)

    # 滑动：每步加入新元素、移除旧元素
    for p in range(1, ps):
        s_new, e_new = frame_bounds(frame, p, ps, ho)
        s_old, e_old = frame_bounds(frame, p - 1, ps, ho)

        # 移除离开帧的元素
        for wp in range(s_old, s_new):
            if 0 <= wp < ps:
                if av is not None:
                    v = av[indices[wp]]
                    if v is not None:
                        window_sum -= v
                        window_count -= 1
                else:
                    window_sum -= 1
                    window_count -= 1

        # 加入进入帧的元素
        for wp in range(e_old + 1, e_new + 1):
            if 0 <= wp < ps:
                if av is not None:
                    v = av[indices[wp]]
                    if v is not None:
                        window_sum += v
                        window_count += 1
                else:
                    window_sum += 1
                    window_count += 1

        _emit(name, indices[p], results, window_sum, window_count)


def _emit(name, idx, results, window_sum, window_count):
    """写入聚合结果。"""
    if name == 'SUM':
        results[idx] = window_sum
    elif name == 'COUNT':
        results[idx] = window_count
    elif name == 'AVG':
        results[idx] = (window_sum / window_count
                        if window_count > 0 else None)


def _can_use_sliding(name, frame, has_nulls):
    """判断是否可用滑动窗口。"""
    if name not in ('SUM', 'COUNT', 'AVG'):
        return False
    if has_nulls and name in ('SUM', 'AVG'):
        # 有 NULL 时滑动窗口的加减可能不精确（NULL 值需特殊处理）
        # 实际上上面的实现已处理 NULL，可以启用
        pass
    if frame is None:
        return False
    # 只有固定偏移帧才能用滑动
    fs = frame.start
    fe = frame.end
    if fs and fs.type in ('N_PRECEDING', 'CURRENT_ROW'):
        if fe and fe.type in ('N_FOLLOWING', 'CURRENT_ROW'):
            return True
    return False