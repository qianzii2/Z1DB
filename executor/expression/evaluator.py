from __future__ import annotations
"""表达式求值器 — 核心类。
辅助方法继承自 helpers.EvalHelpers。
类型推断委托到 type_inference。
函数分发委托到 dispatch_table。"""
import math as _math
import operator as _op
import re as _re
import threading
from typing import Any, Dict, List, Optional

from executor.core.vector import DataVector
from executor.core.batch import VectorBatch
from executor.expression.helpers import EvalHelpers, _cast_to_bool
from executor.expression.type_inference import infer_type, cast_vector
from executor.expression.dispatch_table import build_dispatch
from metal.bitmap import Bitmap
from metal.typed_vector import TypedVector
from parser.ast import (
    AggregateCall, AliasExpr, BetweenExpr, BinaryExpr,
    CaseExpr, CastExpr, ColumnRef, FunctionCall, InExpr,
    IsNullExpr, LikeExpr, Literal, StarExpr, UnaryExpr)
from storage.types import (
    DTYPE_TO_ARRAY_CODE, DataType, is_numeric, is_string,
    promote, resolve_type_name, _NUMERIC_RANK)
from utils.errors import (
    DivisionByZeroError, ExecutionError,
    NumericOverflowError, TypeMismatchError)

try:
    from parser.ast import WindowCall
except ImportError:
    WindowCall = None
try:
    from executor.string_algo.boyer_moore import BoyerMoore
    _HAS_BM = True
except ImportError:
    _HAS_BM = False
try:
    from executor.vectorized_ops import try_vectorized_arith, try_vectorized_cmp_scalar
    _HAS_VEC_OPS = True
except ImportError:
    _HAS_VEC_OPS = False
try:
    from executor.expression.scalar_registry import ScalarFunctionRegistry
    _HAS_SCALAR_REG = True
except ImportError:
    _HAS_SCALAR_REG = False
try:
    from metal.bitmagic import (
        nanbox_batch_eq, nanbox_batch_lt, nanbox_batch_gt,
        nanbox_batch_add, NULL_TAG as _NANBOX_NULL_TAG)
    _HAS_NANBOX_BATCH = True
except ImportError:
    _HAS_NANBOX_BATCH = False

# 线程安全 LIKE/DFA 缓存
_LIKE_LOCK = threading.Lock()
_LIKE_CACHE: Dict[str, _re.Pattern] = {}
_LIKE_MAX = 1024
_DFA_LOCK = threading.Lock()
_DFA_CACHE: Dict[str, Any] = {}
_DFA_MAX = 256


def _like_match(text: str, pattern: str) -> bool:
    with _LIKE_LOCK:
        compiled = _LIKE_CACHE.get(pattern)
    if compiled is None:
        regex = '^'
        for ch in pattern:
            if ch == '%': regex += '.*'
            elif ch == '_': regex += '.'
            elif ch in r'\.^$+?{}[]|()': regex += '\\' + ch
            else: regex += ch
        regex += '$'
        compiled = _re.compile(regex, _re.DOTALL)
        with _LIKE_LOCK:
            if len(_LIKE_CACHE) < _LIKE_MAX:
                _LIKE_CACHE[pattern] = compiled
    return compiled.match(text) is not None


class ExpressionEvaluator(EvalHelpers):
    """表达式求值器。继承 EvalHelpers 获得 40+ 辅助方法。"""

    _FUNC_DISPATCH: Optional[Dict[str, Any]] = None

    def __init__(self, registry: Optional[Any] = None):
        self._registry = registry

    # ═══ 主入口 ═══

    def evaluate(self, expr: Any, batch: VectorBatch) -> DataVector:
        if isinstance(expr, Literal):
            return self._eval_literal(expr, batch.row_count)
        if isinstance(expr, ColumnRef):
            return self._eval_colref(expr, batch)
        if isinstance(expr, AliasExpr):
            return self.evaluate(expr.expr, batch)
        if isinstance(expr, BinaryExpr):
            return self._eval_binary(
                expr.op,
                self.evaluate(expr.left, batch),
                self.evaluate(expr.right, batch))
        if isinstance(expr, UnaryExpr):
            return self._eval_unary(expr, batch)
        if isinstance(expr, IsNullExpr):
            return self._eval_is_null(expr, batch)
        if isinstance(expr, CaseExpr):
            return self._eval_case(expr, batch)
        if isinstance(expr, CastExpr):
            return self._eval_cast(expr, batch)
        if isinstance(expr, InExpr):
            return self._eval_in(expr, batch)
        if isinstance(expr, BetweenExpr):
            return self._eval_between(expr, batch)
        if isinstance(expr, LikeExpr):
            return self._eval_like(expr, batch)
        if isinstance(expr, FunctionCall):
            return self._eval_function(expr, batch)
        if isinstance(expr, AggregateCall):
            raise ExecutionError("聚合出现在非聚合上下文中")
        if isinstance(expr, StarExpr):
            raise ExecutionError("内部错误: StarExpr 在求值阶段")
        if WindowCall and isinstance(expr, WindowCall):
            raise ExecutionError("窗口函数出现在窗口上下文外")
        raise ExecutionError(f"未知表达式: {type(expr).__name__}")

    def evaluate_predicate(self, expr, batch):
        vec = self.evaluate(expr, batch)
        if vec.dtype == DataType.BOOLEAN:
            return self._bool_to_bitmap(vec)
        if is_numeric(vec.dtype):
            bm = Bitmap(len(vec))
            for i in range(len(vec)):
                if not vec.is_null(i) and vec.get(i) != 0:
                    bm.set_bit(i)
            return bm
        raise ExecutionError(f"WHERE 必须是布尔类型，实际为 {vec.dtype.name}")

    # ═══ 列/字面量 ═══

    def _eval_colref(self, expr, batch):
        if expr.table:
            q = f"{expr.table}.{expr.column}"
            if q in batch.columns: return batch.columns[q]
        if expr.column in batch.columns:
            return batch.columns[expr.column]
        if expr.table:
            raise ExecutionError(f"列 '{expr.table}.{expr.column}' 不存在")
        raise ExecutionError(f"列 '{expr.column}' 不存在")

    def _eval_literal(self, lit, n):
        if lit.value is None:
            dt = lit.inferred_type if lit.inferred_type != DataType.UNKNOWN else DataType.INT
            nulls = Bitmap(n)
            for i in range(n): nulls.set_bit(i)
            return DataVector.from_nulls(dt, n, nulls)
        dtype = lit.inferred_type if lit.inferred_type != DataType.UNKNOWN else DataType.INT
        code = DTYPE_TO_ARRAY_CODE.get(dtype)
        if code is not None:
            data: Any = TypedVector(code, [lit.value] * n)
        elif dtype in (DataType.VARCHAR, DataType.TEXT):
            data = [lit.value] * n
        elif dtype == DataType.BOOLEAN:
            data = Bitmap(n)
            if lit.value:
                for i in range(n): data.set_bit(i)
        else:
            data = TypedVector('q', [lit.value] * n)
        return DataVector(dtype=dtype, data=data, nulls=Bitmap(n), _length=n)

    # ═══ 二元运算 ═══

    def _eval_binary(self, op, left, right):
        if op in ('+', '-', '*', '/', '%'):
            return self._eval_arith(op, left, right)
        if op in ('=', '!=', '<', '>', '<=', '>='):
            return self._eval_cmp(op, left, right)
        if op == '||':
            return self._eval_concat(left, right)
        if op == 'AND':
            return self._eval_and(self._to_bool(left), self._to_bool(right))
        if op == 'OR':
            return self._eval_or(self._to_bool(left), self._to_bool(right))
        raise ExecutionError(f"不支持的运算符: {op}")

    def _eval_arith(self, op, left, right):
        """nanbox_batch 批量算术：+, -, * 都走批量路径。"""

        if op in ('+', '-'):
            if self._is_temporal_arith(left, right):
                result = self._eval_temporal_arith(
                    op, left, right, len(left))
                if result is not None:
                    return result
            # 如果 _is_temporal_arith 返回 True 但 _eval_temporal_arith
            # 返回 None（如 int - date），跳过 promote 直接报错
            from storage.types import is_temporal
            if (is_temporal(left.dtype) or is_temporal(right.dtype)):
                raise ExecutionError(
                    f"不支持的时间算术: {left.dtype.name} {op} {right.dtype.name}")

        target = promote(left.dtype, right.dtype)
        lv = cast_vector(left, target)
        rv = cast_vector(right, target)
        n = len(lv)
        is_int = target in (DataType.INT, DataType.BIGINT, DataType.BOOLEAN)
        code = DTYPE_TO_ARRAY_CODE.get(target)
        is_float = target in (DataType.FLOAT, DataType.DOUBLE)

        # nanbox_batch 批量路径：+, -, *
        if _HAS_NANBOX_BATCH and op in ('+', '-', '*'):
            if (lv._packed is not None and rv._packed is not None
                    and len(lv._packed) == n and len(rv._packed) == n):
                try:
                    from executor.core.vector import _unpack_to_typed
                    if op == '+':
                        from metal.bitmagic import nanbox_batch_add
                        rp, nb = nanbox_batch_add(lv._packed, rv._packed, n, is_float)
                    elif op == '-':
                        from metal.bitmagic import nanbox_batch_sub
                        rp, nb = nanbox_batch_sub(lv._packed, rv._packed, n, is_float)
                    else:  # '*'
                        from metal.bitmagic import nanbox_batch_mul
                        rp, nb = nanbox_batch_mul(lv._packed, rv._packed, n, is_float)
                    nn = Bitmap(n)
                    for i in range(n):
                        if nb[i >> 3] & (1 << (i & 7)):
                            nn.set_bit(i)
                    nd = _unpack_to_typed(target, rp, n)
                    return DataVector(dtype=target, data=nd, nulls=nn,
                                      _length=n, _packed=rp)
                except Exception:
                    pass

        # 向量化快速路径（无 NULL + TypedVector）
        if (_HAS_VEC_OPS and op in ('+', '-', '*')
                and isinstance(lv.data, TypedVector)
                and isinstance(rv.data, TypedVector)
                and lv.nulls.popcount() == 0
                and rv.nulls.popcount() == 0):
            ra = try_vectorized_arith(
                op, lv.data.to_array(), rv.data.to_array(),
                n, lv.data.dtype_code)
            if ra is not None:
                nd = TypedVector(lv.data.dtype_code)
                nd._array = ra
                return DataVector(dtype=target, data=nd,
                                  nulls=Bitmap(n), _length=n)

        if op in ('+', '-') and self._is_temporal_arith(lv, rv):
            return self._eval_temporal_arith(op, lv, rv, n)

        # 通用逐行路径
        ops = {'+': _op.add, '-': _op.sub, '*': _op.mul}
        rd: Any = TypedVector(code) if code else []
        rn = Bitmap(n)
        for i in range(n):
            if lv.is_null(i) or rv.is_null(i):
                rn.set_bit(i)
                rd.append(0 if isinstance(rd, TypedVector) else None)
                continue
            a, b = lv.get(i), rv.get(i)
            if op in ops:
                val = ops[op](a, b)
            elif op == '/':
                if b == 0:
                    if is_int:
                        raise DivisionByZeroError()
                    val = float('inf') if a >= 0 else float('-inf')
                else:
                    val = _math.trunc(a / b) if is_int else a / b
            elif op == '%':
                if b == 0:
                    raise DivisionByZeroError()
                val = a % b
            else:
                raise ExecutionError(f"未知算术运算: {op}")
            if target == DataType.INT and not (-2**31 <= val < 2**31):
                raise NumericOverflowError(f"溢出: {val}")
            rd.append(val)
        return DataVector(dtype=target, data=rd, nulls=rn, _length=n)

    def _eval_cmp(self, op, left, right):
        if (op == '=' and left.dict_encoded is not None
                and right._length > 0 and not right.is_null(0)):
            target = right.get(0); de = left.dict_encoded
            code = de.lookup_code(target); n = len(left)
            rd = Bitmap(n); rn = Bitmap(n)
            if code is not None:
                for i in range(n):
                    if left.is_null(i): rn.set_bit(i)
                    elif de.codes[i] == code: rd.set_bit(i)
            return DataVector(dtype=DataType.BOOLEAN, data=rd, nulls=rn, _length=n)

        if is_numeric(left.dtype) and is_numeric(right.dtype):
            t = promote(left.dtype, right.dtype)
            lv = cast_vector(left, t); rv = cast_vector(right, t)
        elif is_string(left.dtype) and is_string(right.dtype):
            lv, rv = left, right
        elif left.dtype == DataType.BOOLEAN and right.dtype == DataType.BOOLEAN:
            lv, rv = left, right
        else:
            try:
                t = promote(left.dtype, right.dtype)
                lv = cast_vector(left, t); rv = cast_vector(right, t)
            except TypeMismatchError:
                lv, rv = left, right

        n = len(lv)
        fns = {'=': _op.eq, '!=': _op.ne, '<': _op.lt, '>': _op.gt, '<=': _op.le, '>=': _op.ge}
        fn = fns[op]; rd = Bitmap(n); rn = Bitmap(n)

        if _HAS_NANBOX_BATCH:
            if (lv._packed is not None and rv._packed is not None
                    and len(lv._packed) == n and len(rv._packed) == n):
                try:
                    if op == '=':
                        rd._data = nanbox_batch_eq(lv._packed, rv._packed, n)
                        rd._logical_size = n
                        return DataVector(dtype=DataType.BOOLEAN, data=rd, nulls=rn, _length=n)
                    if op == '<':
                        rd._data = nanbox_batch_lt(lv._packed, rv._packed, n)
                        rd._logical_size = n
                        return DataVector(dtype=DataType.BOOLEAN, data=rd, nulls=rn, _length=n)
                    if op == '>':
                        rd._data = nanbox_batch_gt(lv._packed, rv._packed, n)
                        rd._logical_size = n
                        return DataVector(dtype=DataType.BOOLEAN, data=rd, nulls=rn, _length=n)
                    if op in ('<=', '>=', '!='):
                        bmp = (nanbox_batch_gt if op == '<=' else nanbox_batch_lt if op == '>=' else nanbox_batch_eq)(lv._packed, rv._packed, n)
                        rd._data = bytearray(len(bmp))
                        for bi in range(len(bmp)): rd._data[bi] = (~bmp[bi]) & 0xFF
                        rd._logical_size = n
                        for i in range(n):
                            if lv._packed[i] == _NANBOX_NULL_TAG or rv._packed[i] == _NANBOX_NULL_TAG:
                                rd.clear_bit(i); rn.set_bit(i)
                        return DataVector(dtype=DataType.BOOLEAN, data=rd, nulls=rn, _length=n)
                except Exception:
                    rd = Bitmap(n); rn = Bitmap(n)

        if (_HAS_VEC_OPS and isinstance(lv.data, TypedVector)
                and lv.nulls.popcount() == 0 and rv.nulls.popcount() == 0
                and rv._length == 1 and n > 1):
            scalar = rv.get(0)
            rb = try_vectorized_cmp_scalar(op, lv.data.to_array(), scalar, n)
            if rb is not None:
                rd._data = rb; rd._logical_size = n
                return DataVector(dtype=DataType.BOOLEAN, data=rd, nulls=rn, _length=n)

        if (isinstance(lv.data, TypedVector) and isinstance(rv.data, TypedVector)
                and lv.nulls.popcount() == 0 and rv.nulls.popcount() == 0):
            la = lv.data.to_array(); ra = rv.data.to_array()
            for i in range(n):
                if fn(la[i], ra[i]): rd.set_bit(i)
            return DataVector(dtype=DataType.BOOLEAN, data=rd, nulls=rn, _length=n)

        if lv.nulls.popcount() > 0 or rv.nulls.popcount() > 0:
            nu = lv.nulls.or_op(rv.nulls)
            for i in range(n):
                if nu.get_bit(i): rn.set_bit(i)
                elif fn(lv.get(i), rv.get(i)): rd.set_bit(i)
        else:
            for i in range(n):
                if fn(lv.get(i), rv.get(i)): rd.set_bit(i)
        return DataVector(dtype=DataType.BOOLEAN, data=rd, nulls=rn, _length=n)

    def _eval_concat(self, left, right):
        n = len(left); rd = []; rn = Bitmap(n)
        for i in range(n):
            if left.is_null(i) or right.is_null(i): rn.set_bit(i); rd.append('')
            else: rd.append(str(left.get(i)) + str(right.get(i)))
        return DataVector(dtype=DataType.VARCHAR, data=rd, nulls=rn, _length=n)

    def _eval_and(self, l, r):
        n = len(l); rd = Bitmap(n); rn = Bitmap(n)
        for i in range(n):
            ln, rnn = l.is_null(i), r.is_null(i)
            lv = l.get(i) if not ln else None; rv = r.get(i) if not rnn else None
            if (not ln and not lv) or (not rnn and not rv): pass
            elif ln or rnn: rn.set_bit(i)
            elif lv and rv: rd.set_bit(i)
        return DataVector(dtype=DataType.BOOLEAN, data=rd, nulls=rn, _length=n)

    def _eval_or(self, l, r):
        n = len(l); rd = Bitmap(n); rn = Bitmap(n)
        for i in range(n):
            ln, rnn = l.is_null(i), r.is_null(i)
            lv = l.get(i) if not ln else None; rv = r.get(i) if not rnn else None
            if (not ln and lv) or (not rnn and rv): rd.set_bit(i)
            elif ln or rnn: rn.set_bit(i)
        return DataVector(dtype=DataType.BOOLEAN, data=rd, nulls=rn, _length=n)

    # ═══ 一元/IS NULL/CASE/CAST ═══

    def _eval_unary(self, expr, batch):
        ov = self.evaluate(expr.operand, batch)
        if expr.op == '+': return ov
        if expr.op == '-':
            n = len(ov)
            if not is_numeric(ov.dtype): raise ExecutionError(f"无法取负: {ov.dtype.name}")
            code = DTYPE_TO_ARRAY_CODE.get(ov.dtype)
            if not code: raise ExecutionError(f"无法取负: {ov.dtype.name}")
            rd = TypedVector(code); rn = Bitmap(n)
            for i in range(n):
                if ov.is_null(i): rn.set_bit(i); rd.append(0)
                else: rd.append(-ov.get(i))
            return DataVector(dtype=ov.dtype, data=rd, nulls=rn, _length=n)
        if expr.op == 'NOT':
            bv = self._to_bool(ov); n = len(bv)
            rd = Bitmap(n); rn = Bitmap(n)
            for i in range(n):
                if bv.is_null(i): rn.set_bit(i)
                elif not bv.data.get_bit(i): rd.set_bit(i)
            return DataVector(dtype=DataType.BOOLEAN, data=rd, nulls=rn, _length=n)
        raise ExecutionError(f"不支持的一元运算: {expr.op}")

    def _eval_is_null(self, expr, batch):
        ov = self.evaluate(expr.expr, batch); n = len(ov)
        rd = Bitmap(n)
        for i in range(n):
            isn = ov.is_null(i)
            if (isn and not expr.negated) or (not isn and expr.negated): rd.set_bit(i)
        return DataVector(dtype=DataType.BOOLEAN, data=rd, nulls=Bitmap(n), _length=n)

    def _eval_case(self, expr, batch):
        n = batch.row_count; results = [None] * n; resolved = [False] * n
        if expr.operand is not None:
            opv = self.evaluate(expr.operand, batch)
            for cond, result in expr.when_clauses:
                cv = self.evaluate(cond, batch); rv = self.evaluate(result, batch)
                for i in range(n):
                    if resolved[i]: continue
                    if not opv.is_null(i) and not cv.is_null(i) and opv.get(i) == cv.get(i):
                        results[i] = rv.get(i); resolved[i] = True
        else:
            for cond, result in expr.when_clauses:
                cv = self.evaluate(cond, batch); rv = self.evaluate(result, batch)
                for i in range(n):
                    if resolved[i]: continue
                    if not cv.is_null(i) and cv.get(i): results[i] = rv.get(i); resolved[i] = True
        if expr.else_expr is not None:
            ev = self.evaluate(expr.else_expr, batch)
            for i in range(n):
                if not resolved[i]: results[i] = ev.get(i)
        cs = {cn: batch.columns[cn].dtype for cn in batch.column_names}
        dtype = DataType.UNKNOWN
        for _, re_ in expr.when_clauses:
            dt = infer_type(re_, cs)
            if dt != DataType.UNKNOWN: dtype = dt; break
        if dtype == DataType.UNKNOWN and expr.else_expr:
            dt = infer_type(expr.else_expr, cs)
            if dt != DataType.UNKNOWN: dtype = dt
        if dtype == DataType.UNKNOWN: dtype = DataType.VARCHAR
        return self._list_to_vec(results, dtype, n)

    def _eval_cast(self, expr, batch):
        src = self.evaluate(expr.expr, batch)
        assert expr.type_name
        tdt, _ = resolve_type_name(expr.type_name.name, expr.type_name.params)
        if src.dtype == tdt: return src
        n = len(src); results = [None] * n
        for i in range(n):
            if src.is_null(i): continue
            v = src.get(i)
            try:
                if tdt == DataType.INT: results[i] = int(v)
                elif tdt == DataType.BIGINT: results[i] = int(v)
                elif tdt in (DataType.FLOAT, DataType.DOUBLE): results[i] = float(v)
                elif tdt in (DataType.VARCHAR, DataType.TEXT): results[i] = str(v)
                elif tdt == DataType.BOOLEAN: results[i] = _cast_to_bool(v)
                else: results[i] = v
            except (ValueError, TypeError): pass
        return self._list_to_vec(results, tdt, n)

    # ═══ IN / BETWEEN / LIKE ═══

    def _eval_in(self, expr, batch):
        src = self.evaluate(expr.expr, batch); n = len(src)
        if all(isinstance(v, Literal) for v in expr.values):
            val_set = set(); has_null_val = False
            for v in expr.values:
                if v.value is None: has_null_val = True
                else: val_set.add(v.value)
            rd = Bitmap(n); rn = Bitmap(n)
            for i in range(n):
                if src.is_null(i): rn.set_bit(i); continue
                sv = src.get(i); found = sv in val_set
                if found:
                    if not expr.negated: rd.set_bit(i)
                elif has_null_val: rn.set_bit(i)
                elif expr.negated: rd.set_bit(i)
            return DataVector(dtype=DataType.BOOLEAN, data=rd, nulls=rn, _length=n)
        vvs = [self.evaluate(v, batch) for v in expr.values]
        rd = Bitmap(n); rn = Bitmap(n)
        for i in range(n):
            if src.is_null(i): rn.set_bit(i); continue
            sv = src.get(i); found = False; has_null = False
            for vv in vvs:
                if vv.is_null(i): has_null = True; continue
                if sv == vv.get(i): found = True; break
            if found:
                if not expr.negated: rd.set_bit(i)
            elif has_null: rn.set_bit(i)
            elif expr.negated: rd.set_bit(i)
        return DataVector(dtype=DataType.BOOLEAN, data=rd, nulls=rn, _length=n)

    def _eval_between(self, expr, batch):
        src = self.evaluate(expr.expr, batch)
        lo = self.evaluate(expr.low, batch); hi = self.evaluate(expr.high, batch)
        n = len(src); rd = Bitmap(n); rn = Bitmap(n)
        for i in range(n):
            if src.is_null(i) or lo.is_null(i) or hi.is_null(i): rn.set_bit(i); continue
            inr = lo.get(i) <= src.get(i) <= hi.get(i)
            if (inr and not expr.negated) or (not inr and expr.negated): rd.set_bit(i)
        return DataVector(dtype=DataType.BOOLEAN, data=rd, nulls=rn, _length=n)

    def _eval_like(self, expr, batch):
        src = self.evaluate(expr.expr, batch); pat = self.evaluate(expr.pattern, batch)
        n = len(src); rd = Bitmap(n); rn = Bitmap(n)
        if _HAS_BM and isinstance(expr.pattern, Literal) and isinstance(expr.pattern.value, str):
            p = expr.pattern.value
            if p.startswith('%') and p.endswith('%') and len(p) >= 6:
                inner = p[1:-1]
                if '%' not in inner and '_' not in inner:
                    bm = BoyerMoore(inner)
                    for i in range(n):
                        if src.is_null(i): rn.set_bit(i)
                        else:
                            m = bm.contains(str(src.get(i)))
                            if (m and not expr.negated) or (not m and expr.negated): rd.set_bit(i)
                    return DataVector(dtype=DataType.BOOLEAN, data=rd, nulls=rn, _length=n)
        for i in range(n):
            if src.is_null(i) or pat.is_null(i): rn.set_bit(i); continue
            m = _like_match(str(src.get(i)), str(pat.get(i)))
            if (m and not expr.negated) or (not m and expr.negated): rd.set_bit(i)
        return DataVector(dtype=DataType.BOOLEAN, data=rd, nulls=rn, _length=n)

    # ═══ 函数调用 ═══

    def _eval_function(self, expr, batch):
        name = expr.name.upper(); n = batch.row_count
        if name == 'IF' and len(expr.args) >= 3: return self._if_fn_lazy(expr, batch)
        if name == 'COALESCE' and expr.args: return self._coalesce_lazy(expr, batch)
        args = [self.evaluate(a, batch) for a in expr.args]
        if _HAS_SCALAR_REG:
            entry = ScalarFunctionRegistry.get(name)
            if entry is not None:
                fn_impl, _ = entry
                try: return fn_impl(args, n, self)
                except Exception: pass
        handler = self._get_dispatch().get(name)
        if handler is not None:
            return handler(self, args, n, expr)  # _safe 内部做参数校验
        raise ExecutionError(f"未知函数: {expr.name}")

    def _if_fn_lazy(self, expr, batch):
        n = batch.row_count; cond = self.evaluate(expr.args[0], batch)
        results = [None] * n; ti = []; fi = []
        for i in range(n):
            if not cond.is_null(i) and cond.get(i): ti.append(i)
            else: fi.append(i)
        if ti:
            tv = self.evaluate(expr.args[1], batch)
            for i in ti: results[i] = tv.get(i)
        if fi:
            ev = self.evaluate(expr.args[2], batch)
            for i in fi: results[i] = ev.get(i)
        cs = {cn: batch.columns[cn].dtype for cn in batch.column_names}
        dt = infer_type(expr.args[1], cs)
        return self._list_to_vec(results, dt if dt != DataType.UNKNOWN else DataType.VARCHAR, n)

    def _coalesce_lazy(self, expr, batch):
        n = batch.row_count; results = [None] * n
        resolved = [False] * n; remaining = n; dt = DataType.UNKNOWN
        for arg_expr in expr.args:
            if remaining == 0: break
            vec = self.evaluate(arg_expr, batch)
            if dt == DataType.UNKNOWN and vec.dtype != DataType.UNKNOWN: dt = vec.dtype
            for i in range(n):
                if not resolved[i] and not vec.is_null(i):
                    results[i] = vec.get(i); resolved[i] = True; remaining -= 1
        return self._list_to_vec(results, dt if dt != DataType.UNKNOWN else DataType.INT, n)

    @classmethod
    def _get_dispatch(cls):
        if cls._FUNC_DISPATCH is not None: return cls._FUNC_DISPATCH
        cls._FUNC_DISPATCH = build_dispatch()
        return cls._FUNC_DISPATCH

    @staticmethod
    def _is_temporal_arith(lv, rv):
        """检查是否为时间算术（date±int 或 timestamp±int）。"""
        from storage.types import is_temporal, is_numeric
        if is_temporal(lv.dtype) and is_numeric(rv.dtype):
            return True
        if is_numeric(lv.dtype) and is_temporal(rv.dtype):
            return True
        return False

    def _eval_temporal_arith(self, op, lv, rv, n):
        """[M07] DATE ± 天数 = DATE，TIMESTAMP ± 微秒数 = TIMESTAMP。"""
        from storage.types import is_temporal, DataType
        from metal.bitmap import Bitmap
        from metal.typed_vector import TypedVector
        from storage.types import DTYPE_TO_ARRAY_CODE

        # 确定哪个是时间列、哪个是数值列
        if is_temporal(lv.dtype):
            time_vec, num_vec = lv, rv
            out_dtype = lv.dtype
        else:
            time_vec, num_vec = rv, lv
            out_dtype = rv.dtype
            if op == '-':
                # int - date 无意义 → 回退到通用路径
                return None

        code = DTYPE_TO_ARRAY_CODE.get(out_dtype)
        rd = TypedVector(code) if code else []
        rn = Bitmap(n)
        for i in range(n):
            if time_vec.is_null(i) or num_vec.is_null(i):
                rn.set_bit(i)
                rd.append(0)
                continue
            tv = time_vec.get(i)
            nv = int(num_vec.get(i))
            if op == '+':
                rd.append(tv + nv)
            else:  # '-'
                rd.append(tv - nv)
        return DataVector(dtype=out_dtype, data=rd, nulls=rn, _length=n)

    # ═══ 委托到 type_inference 模块 ═══

    @staticmethod
    def infer_type(expr, schema):
        return infer_type(expr, schema)

    @staticmethod
    def cast_vector(vec, target):
        return cast_vector(vec, target)
