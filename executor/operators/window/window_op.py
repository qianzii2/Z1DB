from __future__ import annotations
"""窗口函数算子 — 自适应帧计算。
集成FenwickTree/WaveletTree/SortedContainer用于高级窗口优化。"""
import functools
from typing import Any, Dict, List, Optional, Tuple
from executor.core.batch import VectorBatch
from executor.core.operator import Operator
from executor.core.vector import DataVector
from executor.expression.evaluator import ExpressionEvaluator
from metal.bitmap import Bitmap
from metal.typed_vector import TypedVector
from parser.ast import (AggregateCall, FunctionCall, SortKey, StarExpr,
                         WindowCall, WindowFrame)
from storage.types import DTYPE_TO_ARRAY_CODE, DataType

try:
    from structures.segment_tree import SegmentTree, MinSegmentTree, MaxSegmentTree
    _HAS_SEGTREE = True
except ImportError:
    _HAS_SEGTREE = False

try:
    from structures.sparse_table import SparseTableMin, SparseTableMax
    _HAS_SPARSE = True
except ImportError:
    _HAS_SPARSE = False

try:
    from structures.fenwick_tree import FenwickTree
    _HAS_FENWICK = True
except ImportError:
    _HAS_FENWICK = False

try:
    from structures.sorted_container import SortedList
    _HAS_SORTED = True
except ImportError:
    _HAS_SORTED = False

try:
    from structures.wavelet_tree import WaveletTree
    _HAS_WAVELET = True
except ImportError:
    _HAS_WAVELET = False


class WindowOperator(Operator):
    def __init__(self, child: Operator,
                 window_specs: List[Tuple[str, WindowCall]]) -> None:
        super().__init__()
        self.child = child; self.children = [child]
        self._specs = window_specs
        self._evaluator = ExpressionEvaluator()
        self._result: Optional[VectorBatch] = None
        self._emitted = False

    def output_schema(self):
        base = self.child.output_schema()
        return base + [(n, self._infer_window_type(wc)) for n, wc in self._specs]

    def _infer_window_type(self, wc):
        fn = wc.func
        if isinstance(fn, FunctionCall):
            u = fn.name.upper()
            if u in ('ROW_NUMBER','RANK','DENSE_RANK','NTILE'): return DataType.BIGINT
            if u in ('PERCENT_RANK','CUME_DIST'): return DataType.DOUBLE
        if isinstance(fn, AggregateCall):
            if fn.name.upper() == 'COUNT': return DataType.BIGINT
            if fn.name.upper() == 'AVG': return DataType.DOUBLE
            if fn.name.upper() == 'MEDIAN': return DataType.DOUBLE
            if fn.name.upper() in ('PERCENTILE_CONT','APPROX_PERCENTILE'): return DataType.DOUBLE
        return DataType.UNKNOWN

    def open(self):
        self.child.open()
        batches = []
        while True:
            b = self._ensure_batch(self.child.next_batch())
            if b is None: break
            batches.append(b)
        self.child.close()
        if not batches:
            s = self.output_schema()
            self._result = VectorBatch.empty([n for n,_ in s],[t for _,t in s])
            self._emitted = False; return
        merged = VectorBatch.merge(batches)
        n = merged.row_count
        for temp_name, wc in self._specs:
            values = self._compute_window(wc, merged, n)
            dt = self._detect_type(values)
            merged.add_column(temp_name, self._list_to_vec(values, dt, n))
        self._result = merged; self._emitted = False

    def next_batch(self):
        if self._emitted: return None
        self._emitted = True; return self._result

    def close(self): pass

    def _compute_window(self, wc, batch, n):
        pv = [self._evaluator.evaluate(e, batch).to_python_list() for e in wc.partition_by]
        se = [(self._evaluator.evaluate(sk.expr, batch).to_python_list(), sk.direction, sk.nulls) for sk in wc.order_by]
        ho = bool(wc.order_by)
        parts: Dict[tuple, List[int]] = {}; po: List[tuple] = []
        for i in range(n):
            k = tuple(p[i] for p in pv) if pv else ()
            if k not in parts: parts[k] = []; po.append(k)
            parts[k].append(i)
        for k in po:
            if se: parts[k].sort(key=functools.cmp_to_key(lambda i,j: self._cmp_rows(i,j,se)))

        results = [None]*n
        fn = wc.func
        fn_name = (fn.name if isinstance(fn,(FunctionCall,AggregateCall)) else '').upper()

        for k in po:
            idx = parts[k]; ps = len(idx)

            if fn_name == 'ROW_NUMBER':
                for r,i in enumerate(idx,1): results[i] = r
            elif fn_name == 'RANK':
                self._compute_rank(idx, se, results, False)
            elif fn_name == 'DENSE_RANK':
                self._compute_rank(idx, se, results, True)
            elif fn_name == 'NTILE':
                nt = int(fn.args[0].value) if fn.args else 1
                for p,i in enumerate(idx): results[i] = (p*nt)//ps + 1
            elif fn_name == 'PERCENT_RANK':
                self._compute_rank(idx, se, results, False)
                for i in idx: results[i] = (results[i]-1)/max(ps-1,1)
            elif fn_name == 'CUME_DIST':
                for p,i in enumerate(idx):
                    le = p
                    while le+1 < ps and self._rows_equal(idx[le+1],idx[p],se): le += 1
                    results[i] = (le+1)/ps
            elif fn_name in ('LAG','LEAD'):
                off = 1; default = None
                if isinstance(fn,FunctionCall) and len(fn.args)>=2:
                    off = int(self._evaluator.evaluate(fn.args[1],batch).get(0) or 1)
                if isinstance(fn,FunctionCall) and len(fn.args)>=3:
                    default = self._evaluator.evaluate(fn.args[2],batch).get(0)
                av = self._evaluator.evaluate(fn.args[0],batch).to_python_list() if fn.args else None
                for p,i in enumerate(idx):
                    s = p-off if fn_name=='LAG' else p+off
                    results[i] = av[idx[s]] if 0<=s<ps and av else default
            elif fn_name in ('FIRST_VALUE','LAST_VALUE','NTH_VALUE'):
                av = self._evaluator.evaluate(fn.args[0],batch).to_python_list() if fn.args else [None]*n
                for p,i in enumerate(idx):
                    st,en = self._frame_bounds(wc.frame,p,ps,ho)
                    if fn_name == 'FIRST_VALUE': results[i] = av[idx[st]]
                    elif fn_name == 'LAST_VALUE': results[i] = av[idx[en]]
                    elif fn_name == 'NTH_VALUE':
                        nth = int(fn.args[1].value) if len(fn.args)>1 else 1
                        t = st+nth-1
                        results[i] = av[idx[t]] if t<=en else None
            elif isinstance(fn, AggregateCall):
                self._compute_agg_window(fn, wc.frame, idx, batch, results, ho)
        return results

    def _compute_rank(self, idx, se, results, dense):
        if not idx: return
        r = d = 1; results[idx[0]] = 1
        for p in range(1,len(idx)):
            if not self._rows_equal(idx[p],idx[p-1],se):
                if dense: d += 1; r = d
                else: r = p+1
            results[idx[p]] = r

    def _rows_equal(self, i, j, se):
        for v,_,_ in se:
            if v[i] != v[j]: return False
        return True

    def _compute_agg_window(self, fn, frame, indices, batch, results, ho):
        name = fn.name.upper()
        if fn.args and not isinstance(fn.args[0], StarExpr):
            av = self._evaluator.evaluate(fn.args[0], batch).to_python_list()
        else:
            av = None
        ps = len(indices)
        vals = []
        for p in range(ps):
            if av is not None:
                v = av[indices[p]]
                vals.append(v if v is not None else 0)
            else:
                vals.append(1)

        fst = frame.start.type if frame and frame.start else 'UNBOUNDED_PRECEDING'
        fen = frame.end.type if frame and frame.end else 'CURRENT_ROW'

        # 无frame + 无ORDER BY → 整个分区
        if frame is None and not ho:
            self._agg_whole(name, vals, indices, results, av, ps); return
        # 无frame + 有ORDER BY 或 UNBOUNDED..CURRENT → 前缀
        if frame is None and ho:
            self._agg_prefix_adaptive(name, vals, indices, results, av, ps); return
        if fst == 'UNBOUNDED_PRECEDING' and fen == 'CURRENT_ROW':
            self._agg_prefix_adaptive(name, vals, indices, results, av, ps); return

        # 固定帧
        fs = frame.start.offset if frame and frame.start and frame.start.type == 'N_PRECEDING' else None
        fe = frame.end.offset if frame and frame.end and frame.end.type == 'N_FOLLOWING' else None
        if fs is not None or fe is not None:
            fsz = (fs or 0) + (fe or 0) + 1
            # MEDIAN / PERCENTILE → WaveletTree
            if name in ('MEDIAN','PERCENTILE_CONT','APPROX_PERCENTILE') and _HAS_WAVELET and ps > 16:
                self._agg_wavelet_percentile(name, vals, indices, results, frame, ps, ho); return
            # MIN/MAX → SparseTable
            if name in ('MIN','MAX') and _HAS_SPARSE and ps > 16:
                self._agg_sparse_table(name, vals, indices, results, frame, ps, ho); return
            # SUM/COUNT/AVG → SegmentTree or FenwickTree
            if name in ('SUM','AVG','COUNT') and ps > 16:
                if _HAS_FENWICK:
                    self._agg_fenwick(name, vals, indices, results, frame, ps, ho); return
                if _HAS_SEGTREE:
                    self._agg_segment_tree(name, vals, indices, results, frame, ps, ho); return
            # 小帧暴力
            if fsz <= 8:
                self._agg_brute(name, vals, indices, results, av, frame, ps, ho); return
            # MEDIAN 滑动窗口 → SortedContainer Mo's
            if name == 'MEDIAN' and _HAS_SORTED and ps > 16:
                self._agg_mos_median(vals, indices, results, frame, ps, ho); return

        self._agg_brute(name, vals, indices, results, av, frame, ps, ho)

    def _agg_whole(self, name, vals, indices, results, av, ps):
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
            a = t/c if c > 0 else None
            for i in indices: results[i] = a
        elif name == 'MIN':
            nn = [vals[p] for p in range(ps) if av is None or av[indices[p]] is not None]
            m = min(nn) if nn else None
            for i in indices: results[i] = m
        elif name == 'MAX':
            nn = [vals[p] for p in range(ps) if av is None or av[indices[p]] is not None]
            m = max(nn) if nn else None
            for i in indices: results[i] = m

    def _agg_prefix_adaptive(self, name, vals, indices, results, av, ps):
        """前缀累加，SUM用FenwickTree加速。"""
        if name == 'SUM' and _HAS_FENWICK and ps > 100:
            ft = FenwickTree.from_list(vals)
            for p, i in enumerate(indices):
                results[i] = ft.prefix_sum(p)
            return
        # 回退普通前缀
        if name == 'COUNT':
            c = 0
            for p,i in enumerate(indices):
                if av is None or av[i] is not None: c += 1
                results[i] = c
        elif name == 'SUM':
            s = 0
            for p,i in enumerate(indices):
                if av is None or av[i] is not None: s += vals[p]
                results[i] = s
        elif name == 'AVG':
            s = c = 0
            for p,i in enumerate(indices):
                if av is None or av[i] is not None: s += vals[p]; c += 1
                results[i] = s/c if c > 0 else None
        elif name == 'MIN':
            cm = None
            for p,i in enumerate(indices):
                if av is None or av[i] is not None:
                    if cm is None or vals[p] < cm: cm = vals[p]
                results[i] = cm
        elif name == 'MAX':
            cm = None
            for p,i in enumerate(indices):
                if av is None or av[i] is not None:
                    if cm is None or vals[p] > cm: cm = vals[p]
                results[i] = cm

    def _agg_fenwick(self, name, vals, indices, results, frame, ps, ho):
        """FenwickTree: O(log n)范围和查询。"""
        ft = FenwickTree.from_list(vals)
        for p in range(ps):
            s, e = self._frame_bounds(frame, p, ps, ho)
            rs = ft.range_sum(s, e)
            if name == 'SUM': results[indices[p]] = rs
            elif name == 'COUNT': results[indices[p]] = e-s+1
            elif name == 'AVG':
                c = e-s+1
                results[indices[p]] = rs/c if c > 0 else None

    def _agg_segment_tree(self, name, vals, indices, results, frame, ps, ho):
        st = SegmentTree(vals)
        for p in range(ps):
            s, e = self._frame_bounds(frame, p, ps, ho)
            rs = st.query(s, e)
            if name == 'SUM': results[indices[p]] = rs
            elif name == 'COUNT': results[indices[p]] = e-s+1
            elif name == 'AVG':
                c = e-s+1
                results[indices[p]] = rs/c if c > 0 else None

    def _agg_sparse_table(self, name, vals, indices, results, frame, ps, ho):
        st = SparseTableMin(vals) if name == 'MIN' else SparseTableMax(vals)
        for p in range(ps):
            s, e = self._frame_bounds(frame, p, ps, ho)
            results[indices[p]] = st.query(s, e)

    def _agg_wavelet_percentile(self, name, vals, indices, results, frame, ps, ho):
        """WaveletTree: O(log σ)范围分位数。"""
        # 将float值映射为整数（保持顺序）
        sorted_unique = sorted(set(vals))
        val_to_int = {v: i for i, v in enumerate(sorted_unique)}
        int_vals = [val_to_int[v] for v in vals]
        sigma = len(sorted_unique)
        if sigma == 0: return

        wt = WaveletTree(int_vals, sigma)
        for p in range(ps):
            s, e = self._frame_bounds(frame, p, ps, ho)
            count = e - s + 1
            if count <= 0:
                results[indices[p]] = None; continue
            # 中位数 = 第 count//2 小的元素
            if name == 'MEDIAN':
                if count % 2 == 1:
                    k = count // 2
                    int_val = wt.quantile(s, e + 1, k)
                    results[indices[p]] = sorted_unique[int_val] if 0 <= int_val < sigma else None
                else:
                    k1 = count // 2 - 1; k2 = count // 2
                    iv1 = wt.quantile(s, e + 1, k1)
                    iv2 = wt.quantile(s, e + 1, k2)
                    if 0 <= iv1 < sigma and 0 <= iv2 < sigma:
                        results[indices[p]] = (sorted_unique[iv1] + sorted_unique[iv2]) / 2
                    else:
                        results[indices[p]] = None
            else:  # PERCENTILE_CONT etc
                k = max(0, min(count - 1, count // 2))
                int_val = wt.quantile(s, e + 1, k)
                results[indices[p]] = sorted_unique[int_val] if 0 <= int_val < sigma else None

    def _agg_mos_median(self, vals, indices, results, frame, ps, ho):
        """Mo's算法 + SortedContainer: O(n√n)滑动窗口中位数。"""
        sl = SortedList()
        cur_s = cur_e = 0
        for p in range(ps):
            s, e = self._frame_bounds(frame, p, ps, ho)
            # 增量调整窗口
            while cur_e <= e:
                sl.add(vals[cur_e]); cur_e += 1
            while cur_s < s:
                sl.remove(vals[cur_s]); cur_s += 1
            # 有时需要向左扩展
            while cur_s > s:
                cur_s -= 1; sl.add(vals[cur_s])
            results[indices[p]] = sl.median()

    def _agg_brute(self, name, vals, indices, results, av, frame, ps, ho):
        for p in range(ps):
            s, e = self._frame_bounds(frame, p, ps, ho)
            wv = []
            for wp in range(s, e+1):
                if av is not None:
                    v = av[indices[wp]]
                    if v is not None: wv.append(v)
                else:
                    wv.append(1)
            if name == 'COUNT': results[indices[p]] = len(wv)
            elif name == 'SUM': results[indices[p]] = sum(wv) if wv else None
            elif name == 'AVG': results[indices[p]] = sum(wv)/len(wv) if wv else None
            elif name == 'MIN': results[indices[p]] = min(wv) if wv else None
            elif name == 'MAX': results[indices[p]] = max(wv) if wv else None
            elif name == 'MEDIAN':
                if wv:
                    wv.sort(); m = len(wv)
                    results[indices[p]] = wv[m//2] if m%2==1 else (wv[m//2-1]+wv[m//2])/2
                else: results[indices[p]] = None

    def _frame_bounds(self, frame, pos, psize, ho=True):
        if frame is None:
            return (0, pos) if ho else (0, psize-1)
        s, e = 0, pos
        if frame.start:
            if frame.start.type == 'UNBOUNDED_PRECEDING': s = 0
            elif frame.start.type == 'CURRENT_ROW': s = pos
            elif frame.start.type == 'N_PRECEDING': s = max(0, pos-(frame.start.offset or 0))
            elif frame.start.type == 'N_FOLLOWING': s = min(psize-1, pos+(frame.start.offset or 0))
        if frame.end:
            if frame.end.type == 'UNBOUNDED_FOLLOWING': e = psize-1
            elif frame.end.type == 'CURRENT_ROW': e = pos
            elif frame.end.type == 'N_PRECEDING': e = max(0, pos-(frame.end.offset or 0))
            elif frame.end.type == 'N_FOLLOWING': e = min(psize-1, pos+(frame.end.offset or 0))
        return s, e

    def _cmp_rows(self, i, j, se):
        for v,d,np in se:
            np = np or ('NULLS_LAST' if d=='ASC' else 'NULLS_FIRST')
            a, b = v[i], v[j]
            if a is None and b is None: continue
            if a is None: return 1 if np=='NULLS_LAST' else -1
            if b is None: return -1 if np=='NULLS_LAST' else 1
            if a < b: c = -1
            elif a > b: c = 1
            else: continue
            return -c if d=='DESC' else c
        return 0

    def _detect_type(self, values):
        for v in values:
            if v is None: continue
            if isinstance(v, bool): return DataType.BOOLEAN
            if isinstance(v, int): return DataType.BIGINT
            if isinstance(v, float): return DataType.DOUBLE
            if isinstance(v, str): return DataType.VARCHAR
        return DataType.BIGINT

    def _list_to_vec(self, values, dtype, n):
        if dtype == DataType.UNKNOWN: dtype = DataType.BIGINT
        code = DTYPE_TO_ARRAY_CODE.get(dtype); nulls = Bitmap(n)
        if dtype in (DataType.VARCHAR, DataType.TEXT):
            data: Any = ['' if v is None else str(v) for v in values]
            for i in range(n):
                if values[i] is None: nulls.set_bit(i)
        elif dtype == DataType.BOOLEAN:
            data = Bitmap(n)
            for i in range(n):
                if values[i] is None: nulls.set_bit(i)
                elif values[i]: data.set_bit(i)
        elif code:
            data = TypedVector(code)
            for i in range(n):
                if values[i] is None: nulls.set_bit(i); data.append(0)
                else: data.append(values[i])
        else:
            data = TypedVector('q')
            for i in range(n):
                if values[i] is None: nulls.set_bit(i); data.append(0)
                else: data.append(int(values[i]))
        return DataVector(dtype=dtype, data=data, nulls=nulls, _length=n)
