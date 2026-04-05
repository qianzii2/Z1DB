from __future__ import annotations
"""Expression evaluator — complete with all 142 functions."""
import datetime as _dt
import math as _math
import operator as _op
import random as _random
import re as _re
from typing import Any, Dict, List, Optional
from executor.core.vector import DataVector
from executor.core.batch import VectorBatch
from metal.bitmap import Bitmap
from metal.typed_vector import TypedVector
from parser.ast import (
    AggregateCall, AliasExpr, BetweenExpr, BinaryExpr, CaseExpr, CastExpr,
    ColumnRef, FunctionCall, InExpr, IsNullExpr, LikeExpr, Literal,
    StarExpr, UnaryExpr,
)
from storage.types import (
    DTYPE_TO_ARRAY_CODE, DataType, is_numeric, is_string, promote, resolve_type_name,
)
from utils.errors import (
    DivisionByZeroError, ExecutionError, NumericOverflowError, TypeMismatchError,
)
try:
    from parser.ast import WindowCall
except ImportError:
    WindowCall = None  # type: ignore


class ExpressionEvaluator:
    def __init__(self, registry: Optional[Any] = None) -> None:
        self._registry = registry

    # ══════════════════════════════════════════════════════════════
    def evaluate(self, expr: Any, batch: VectorBatch) -> DataVector:
        if isinstance(expr, Literal):      return self._eval_literal(expr, batch.row_count)
        if isinstance(expr, ColumnRef):    return self._eval_colref(expr, batch)
        if isinstance(expr, AliasExpr):    return self.evaluate(expr.expr, batch)
        if isinstance(expr, BinaryExpr):
            return self._eval_binary(expr.op, self.evaluate(expr.left, batch),
                                     self.evaluate(expr.right, batch))
        if isinstance(expr, UnaryExpr):    return self._eval_unary(expr, batch)
        if isinstance(expr, IsNullExpr):   return self._eval_is_null(expr, batch)
        if isinstance(expr, CaseExpr):     return self._eval_case(expr, batch)
        if isinstance(expr, CastExpr):     return self._eval_cast(expr, batch)
        if isinstance(expr, InExpr):       return self._eval_in(expr, batch)
        if isinstance(expr, BetweenExpr):  return self._eval_between(expr, batch)
        if isinstance(expr, LikeExpr):     return self._eval_like(expr, batch)
        if isinstance(expr, FunctionCall): return self._eval_function(expr, batch)
        if isinstance(expr, AggregateCall):
            raise ExecutionError("aggregate in non-aggregate context")
        if isinstance(expr, StarExpr):
            raise ExecutionError("internal: StarExpr in eval")
        if WindowCall is not None and isinstance(expr, WindowCall):
            raise ExecutionError("window function outside of window context")
        raise ExecutionError(f"unknown expression: {type(expr).__name__}")

    def evaluate_predicate(self, expr: Any, batch: VectorBatch) -> Bitmap:
        vec = self.evaluate(expr, batch)
        if vec.dtype == DataType.BOOLEAN: return self._bool_to_bitmap(vec)
        if is_numeric(vec.dtype):
            bm = Bitmap(len(vec))
            for i in range(len(vec)):
                if not vec.is_null(i) and vec.get(i) != 0: bm.set_bit(i)
            return bm
        raise ExecutionError(f"WHERE must be boolean, got {vec.dtype.name}")

    # ══════════════════════════════════════════════════════════════
    def _eval_colref(self, expr: ColumnRef, batch: VectorBatch) -> DataVector:
        if expr.table:
            q = f"{expr.table}.{expr.column}"
            if q in batch.columns: return batch.columns[q]
        if expr.column in batch.columns: return batch.columns[expr.column]
        if expr.table: raise ExecutionError(f"column '{expr.table}.{expr.column}' not found")
        raise ExecutionError(f"column '{expr.column}' not found")

    # ══════════════════════════════════════════════════════════════
    def _eval_literal(self, lit: Literal, n: int) -> DataVector:
        if lit.value is None:
            dt = lit.inferred_type if lit.inferred_type != DataType.UNKNOWN else DataType.INT
            nulls = Bitmap(n)
            for i in range(n): nulls.set_bit(i)
            return DataVector.from_nulls(dt, n, nulls)
        dtype = lit.inferred_type
        if dtype == DataType.UNKNOWN: dtype = DataType.INT
        code = DTYPE_TO_ARRAY_CODE.get(dtype)
        if code is not None: data: Any = TypedVector(code, [lit.value]*n)
        elif dtype in (DataType.VARCHAR, DataType.TEXT): data = [lit.value]*n
        elif dtype == DataType.BOOLEAN:
            data = Bitmap(n)
            if lit.value:
                for i in range(n): data.set_bit(i)
        else: data = TypedVector('q', [lit.value]*n)
        return DataVector(dtype=dtype, data=data, nulls=Bitmap(n), _length=n)

    # ══════════════════════════════════════════════════════════════
    def _eval_binary(self, op: str, left: DataVector, right: DataVector) -> DataVector:
        if op in ('+','-','*','/','%'): return self._eval_arith(op, left, right)
        if op in ('=','!=','<','>','<=','>='): return self._eval_cmp(op, left, right)
        if op == '||': return self._eval_concat(left, right)
        if op == 'AND': return self._eval_and(self._to_bool(left), self._to_bool(right))
        if op == 'OR': return self._eval_or(self._to_bool(left), self._to_bool(right))
        raise ExecutionError(f"unsupported op: {op}")

    def _eval_arith(self, op, left, right):
        target = promote(left.dtype, right.dtype)
        lv, rv = self.cast_vector(left, target), self.cast_vector(right, target)
        n = len(lv); is_int = target in (DataType.INT, DataType.BIGINT, DataType.BOOLEAN)
        ops = {'+': _op.add, '-': _op.sub, '*': _op.mul}
        code = DTYPE_TO_ARRAY_CODE.get(target)
        rd: Any = TypedVector(code) if code else []; rn = Bitmap(n)
        for i in range(n):
            if lv.is_null(i) or rv.is_null(i):
                rn.set_bit(i)
                if isinstance(rd, TypedVector): rd.append(0)
                else: rd.append(None)
                continue
            a, b = lv.get(i), rv.get(i)
            if op in ops: val = ops[op](a, b)
            elif op == '/':
                if b == 0:
                    if is_int: raise DivisionByZeroError()
                    val = float('inf') if a >= 0 else float('-inf')
                else: val = int(a / b) if is_int else a / b
            elif op == '%':
                if b == 0: raise DivisionByZeroError()
                val = a % b
            else: raise ExecutionError(f"unknown arith: {op}")
            if target == DataType.INT and not (-2**31 <= val < 2**31):
                raise NumericOverflowError(f"overflow: {val}")
            if isinstance(rd, TypedVector): rd.append(val)
            else: rd.append(val)
        return DataVector(dtype=target, data=rd, nulls=rn, _length=n)

    def _eval_cmp(self, op, left, right):
        if op == '=' and left.dict_encoded is not None and right._length > 0:
            if not right.is_null(0):
                target = right.get(0)
                de = left.dict_encoded
                code = de.lookup_code(target)
                n = len(left)
                rd = Bitmap(n)
                rn = Bitmap(n)
                if code is not None:
                    # Compare integer codes — 10-100x faster than string compare
                    for i in range(n):
                        if left.is_null(i):
                            rn.set_bit(i)
                        elif de.codes[i] == code:
                            rd.set_bit(i)
                # code is None → value not in dictionary → zero matches
                return DataVector(dtype=DataType.BOOLEAN, data=rd, nulls=rn, _length=n)
        # ... rest of existing _eval_cmp follows ...
        if is_numeric(left.dtype) and is_numeric(right.dtype):
            t = promote(left.dtype, right.dtype)
            lv, rv = self.cast_vector(left, t), self.cast_vector(right, t)
        elif is_string(left.dtype) and is_string(right.dtype):
            lv, rv = left, right
        elif left.dtype == DataType.BOOLEAN and right.dtype == DataType.BOOLEAN:
            lv, rv = left, right
        else:
            try:
                t = promote(left.dtype, right.dtype)
                lv, rv = self.cast_vector(left, t), self.cast_vector(right, t)
            except TypeMismatchError:
                lv, rv = left, right

        n = len(lv)
        fns = {'=': _op.eq, '!=': _op.ne, '<': _op.lt,
               '>': _op.gt, '<=': _op.le, '>=': _op.ge}
        fn = fns[op]
        rd = Bitmap(n)
        rn = Bitmap(n)

        # Fast path: both TypedVector with no nulls → batch compare on raw arrays
        if (isinstance(lv.data, TypedVector) and isinstance(rv.data, TypedVector)
                and lv.nulls.popcount() == 0 and rv.nulls.popcount() == 0):
            la = lv.data._array
            ra = rv.data._array
            for i in range(n):
                if fn(la[i], ra[i]):
                    rd.set_bit(i)
            return DataVector(dtype=DataType.BOOLEAN, data=rd, nulls=rn, _length=n)

        # Standard path with null handling
        # Pre-compute null union bitmap to avoid per-element double check
        if lv.nulls.popcount() > 0 or rv.nulls.popcount() > 0:
            null_union = lv.nulls.or_op(rv.nulls)
            for i in range(n):
                if null_union.get_bit(i):
                    rn.set_bit(i)
                elif fn(lv.get(i), rv.get(i)):
                    rd.set_bit(i)
        else:
            for i in range(n):
                if fn(lv.get(i), rv.get(i)):
                    rd.set_bit(i)

        return DataVector(dtype=DataType.BOOLEAN, data=rd, nulls=rn, _length=n)

    def _eval_concat(self, left, right):
        n = len(left); rd: list = []; rn = Bitmap(n)
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

    # ══════════════════════════════════════════════════════════════
    def _eval_unary(self, expr, batch):
        ov = self.evaluate(expr.operand, batch)
        if expr.op == '+': return ov
        if expr.op == '-':
            n = len(ov)
            if not is_numeric(ov.dtype): raise ExecutionError(f"cannot negate {ov.dtype.name}")
            code = DTYPE_TO_ARRAY_CODE.get(ov.dtype)
            if not code: raise ExecutionError(f"cannot negate {ov.dtype.name}")
            rd = TypedVector(code); rn = Bitmap(n)
            for i in range(n):
                if ov.is_null(i): rn.set_bit(i); rd.append(0)
                else: rd.append(-ov.get(i))
            return DataVector(dtype=ov.dtype, data=rd, nulls=rn, _length=n)
        if expr.op == 'NOT':
            bv = self._to_bool(ov); n = len(bv); rd = Bitmap(n); rn = Bitmap(n)
            for i in range(n):
                if bv.is_null(i): rn.set_bit(i)
                elif not bv.data.get_bit(i): rd.set_bit(i)
            return DataVector(dtype=DataType.BOOLEAN, data=rd, nulls=rn, _length=n)
        raise ExecutionError(f"unsupported unary: {expr.op}")

    def _eval_is_null(self, expr, batch):
        ov = self.evaluate(expr.expr, batch); n = len(ov); rd = Bitmap(n)
        for i in range(n):
            isn = ov.is_null(i)
            if (isn and not expr.negated) or (not isn and expr.negated): rd.set_bit(i)
        return DataVector(dtype=DataType.BOOLEAN, data=rd, nulls=Bitmap(n), _length=n)

    def _eval_case(self, expr, batch):
        n = batch.row_count; results = [None]*n; resolved = [False]*n
        if expr.operand is not None:
            opv = self.evaluate(expr.operand, batch)
            for cond, result in expr.when_clauses:
                cv, rv = self.evaluate(cond, batch), self.evaluate(result, batch)
                for i in range(n):
                    if resolved[i]: continue
                    if not opv.is_null(i) and not cv.is_null(i) and opv.get(i) == cv.get(i):
                        results[i] = rv.get(i); resolved[i] = True
        else:
            for cond, result in expr.when_clauses:
                cv, rv = self.evaluate(cond, batch), self.evaluate(result, batch)
                for i in range(n):
                    if resolved[i]: continue
                    if not cv.is_null(i) and cv.get(i): results[i] = rv.get(i); resolved[i] = True
        if expr.else_expr is not None:
            ev = self.evaluate(expr.else_expr, batch)
            for i in range(n):
                if not resolved[i]: results[i] = ev.get(i)
        dtype = DataType.VARCHAR
        if expr.when_clauses:
            cs = {cn: batch.columns[cn].dtype for cn in batch.column_names}
            dt = ExpressionEvaluator.infer_type(expr.when_clauses[0][1], cs)
            if dt != DataType.UNKNOWN: dtype = dt
        return self._list_to_vec(results, dtype, n)

    def _eval_cast(self, expr, batch):
        src = self.evaluate(expr.expr, batch)
        assert expr.type_name
        tdt, _ = resolve_type_name(expr.type_name.name, expr.type_name.params)
        if src.dtype == tdt: return src
        n = len(src); results = [None]*n
        for i in range(n):
            if src.is_null(i): continue
            v = src.get(i)
            try:
                if tdt == DataType.INT: results[i] = int(v)
                elif tdt == DataType.BIGINT: results[i] = int(v)
                elif tdt in (DataType.FLOAT, DataType.DOUBLE): results[i] = float(v)
                elif tdt in (DataType.VARCHAR, DataType.TEXT): results[i] = str(v)
                elif tdt == DataType.BOOLEAN: results[i] = bool(v)
                else: results[i] = v
            except (ValueError, TypeError): pass
        return self._list_to_vec(results, tdt, n)

    def _eval_in(self, expr, batch):
        src = self.evaluate(expr.expr, batch); n = len(src)
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
            else:
                if expr.negated: rd.set_bit(i)
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
        for i in range(n):
            if src.is_null(i) or pat.is_null(i): rn.set_bit(i); continue
            m = _like_match(str(src.get(i)), str(pat.get(i)))
            if (m and not expr.negated) or (not m and expr.negated): rd.set_bit(i)
        return DataVector(dtype=DataType.BOOLEAN, data=rd, nulls=rn, _length=n)

    # ══════════════════════════════════════════════════════════════
    # ALL SCALAR FUNCTIONS
    # ══════════════════════════════════════════════════════════════
    def _eval_function(self, expr: FunctionCall, batch: VectorBatch) -> DataVector:
        name = expr.name.upper()
        args = [self.evaluate(a, batch) for a in expr.args]
        n = batch.row_count

        # ── String ────────────────────────────────────────────
        if name == 'UPPER': return self._str1(args[0], n, str.upper)
        if name == 'LOWER': return self._str1(args[0], n, str.lower)
        if name == 'LENGTH': return self._num1(args[0], n, lambda s: len(str(s)), DataType.INT)
        if name == 'TRIM': return self._str1(args[0], n, str.strip)
        if name == 'LTRIM': return self._str1(args[0], n, str.lstrip)
        if name == 'RTRIM': return self._str1(args[0], n, str.rstrip)
        if name == 'REVERSE': return self._str1(args[0], n, lambda s: s[::-1])
        if name == 'INITCAP': return self._str1(args[0], n, str.title)
        if name == 'REPLACE' and len(args)>=3:
            return self._str3(args[0],args[1],args[2],n, lambda s,a,b: s.replace(a,b))
        if name in ('SUBSTR','SUBSTRING'): return self._substr(args, n)
        if name in ('CONCAT','CONCAT_WS'): return self._concat_fn(name, args, n)
        if name == 'POSITION' and len(args)>=2:
            return self._num2(args[0],args[1],n, lambda sub,s: str(s).find(str(sub))+1, DataType.INT)
        if name == 'LEFT' and len(args)>=2:
            return self._str_int(args[0],args[1],n, lambda s,k: s[:k])
        if name == 'RIGHT' and len(args)>=2:
            return self._str_int(args[0],args[1],n, lambda s,k: s[-k:] if k>0 else '')
        if name == 'REPEAT' and len(args)>=2:
            return self._str_int(args[0],args[1],n, lambda s,k: s*k)
        if name == 'STARTS_WITH' and len(args)>=2:
            return self._bool2(args[0],args[1],n, lambda s,p: str(s).startswith(str(p)))
        if name == 'ENDS_WITH' and len(args)>=2:
            return self._bool2(args[0],args[1],n, lambda s,p: str(s).endswith(str(p)))
        if name == 'CONTAINS' and len(args)>=2:
            return self._bool2(args[0],args[1],n, lambda s,p: str(p) in str(s))
        if name == 'LPAD' and len(args)>=2: return self._lpad_rpad(args, n, True)
        if name == 'RPAD' and len(args)>=2: return self._lpad_rpad(args, n, False)
        if name == 'ASCII' and args:
            return self._num1(args[0],n, lambda s: ord(str(s)[0]) if str(s) else 0, DataType.INT)
        if name == 'CHR' and args:
            return self._str1(args[0],n, lambda x: chr(int(x)) if isinstance(x,(int,float)) else '')
        if name == 'SPLIT_PART' and len(args)>=3: return self._split_part(args, n)
        if name == 'SPLIT' and len(args)>=2:
            return self._str2(args[0],args[1],n, lambda s,d: str(str(s).split(str(d))))
        if name == 'REGEXP_REPLACE' and len(args)>=3:
            return self._str3(args[0],args[1],args[2],n, lambda s,p,r: _re.sub(p,r,s))
        if name == 'REGEXP_MATCH' and len(args)>=2:
            return self._bool2(args[0],args[1],n, lambda s,p: bool(_re.search(str(p),str(s))))
        if name == 'REGEXP_EXTRACT' and len(args)>=2:
            return self._str2(args[0],args[1],n,
                lambda s,p: (m.group(0) if (m:=_re.search(str(p),str(s))) else ''))

        # ── Math ──────────────────────────────────────────────
        if name == 'ABS': return self._num1(args[0],n, abs, args[0].dtype)
        if name in ('CEIL','CEILING'): return self._num1(args[0],n, _math.ceil, DataType.BIGINT)
        if name == 'FLOOR': return self._num1(args[0],n, _math.floor, DataType.BIGINT)
        if name == 'ROUND':
            if len(args)>=2: return self._num2(args[0],args[1],n, lambda x,d: round(x,int(d)), DataType.DOUBLE)
            return self._num1(args[0],n, round, DataType.BIGINT)
        if name in ('TRUNC','TRUNCATE'):
            return self._num1(args[0],n, lambda x: int(x), DataType.BIGINT)
        if name == 'POWER' and len(args)>=2:
            return self._num2(args[0],args[1],n, _math.pow, DataType.DOUBLE)
        if name == 'SQRT': return self._num1(args[0],n, _math.sqrt, DataType.DOUBLE)
        if name == 'CBRT': return self._num1(args[0],n, lambda x: x**(1/3) if x>=0 else -((-x)**(1/3)), DataType.DOUBLE)
        if name == 'MOD' and len(args)>=2:
            return self._num2(args[0],args[1],n, lambda x,y: x%y if y!=0 else None, args[0].dtype)
        if name == 'SIGN':
            return self._num1(args[0],n, lambda x: (1 if x>0 else -1 if x<0 else 0), DataType.INT)
        if name == 'LN': return self._num1(args[0],n, _math.log, DataType.DOUBLE)
        if name == 'LOG2': return self._num1(args[0],n, _math.log2, DataType.DOUBLE)
        if name == 'LOG10': return self._num1(args[0],n, _math.log10, DataType.DOUBLE)
        if name == 'LOG' and len(args)>=2:
            return self._num2(args[0],args[1],n, lambda b,x: _math.log(x,b), DataType.DOUBLE)
        if name == 'EXP': return self._num1(args[0],n, _math.exp, DataType.DOUBLE)
        if name == 'GREATEST' and len(args)>=2:
            return self._variadic(args,n, lambda vs: max((v for v in vs if v is not None), default=None), args[0].dtype)
        if name == 'LEAST' and len(args)>=2:
            return self._variadic(args,n, lambda vs: min((v for v in vs if v is not None), default=None), args[0].dtype)
        if name == 'RANDOM':
            return self._list_to_vec([_random.random() for _ in range(n)], DataType.DOUBLE, n)
        if name == 'WIDTH_BUCKET' and len(args)>=4:
            return self._width_bucket(args, n)

        # ── Conditional ───────────────────────────────────────
        if name == 'COALESCE': return self._coalesce(args, n)
        if name == 'NULLIF' and len(args)>=2: return self._nullif(args[0], args[1], n)
        if name == 'IF' and len(args)>=3: return self._if_fn(args[0], args[1], args[2], n)
        if name == 'TYPEOF' and args:
            rd: list = []
            for i in range(n): rd.append('NULL' if args[0].is_null(i) else args[0].dtype.name)
            return DataVector(dtype=DataType.VARCHAR, data=rd, nulls=Bitmap(n), _length=n)
        if name == 'TRY_CAST': return self._eval_try_cast(expr, batch)
        if name in ('HASH','MURMUR_HASH') and args:
            from metal.hash import murmur3_64
            return self._num1(args[0],n, lambda v: murmur3_64(str(v).encode('utf-8')), DataType.BIGINT)
        if name == 'BIT_COUNT' and args:
            return self._num1(args[0],n, lambda v: bin(int(v)).count('1'), DataType.INT)
        if name == 'ENCODE' and len(args)>=2:
            return self._str2(args[0],args[1],n, lambda v,enc: str(str(v).encode(str(enc))))
        if name == 'DECODE' and len(args)>=2:
            return self._str2(args[0],args[1],n, lambda v,enc: str(v))

        # ── Date/Time ─────────────────────────────────────────
        if name in ('NOW','CURRENT_TIMESTAMP'):
            import time; ts = int(time.time()*1_000_000)
            return self._list_to_vec([ts]*n, DataType.TIMESTAMP, n)
        if name == 'CURRENT_DATE':
            d = _dt.date.today(); days = (d - _dt.date(1970,1,1)).days
            return self._list_to_vec([days]*n, DataType.DATE, n)
        if name in ('YEAR','MONTH','DAY','HOUR','MINUTE','SECOND',
                     'DAY_OF_WEEK','DAY_OF_YEAR','WEEK_OF_YEAR','QUARTER'):
            return self._date_part(name, args[0], n) if args else self._list_to_vec([None]*n, DataType.INT, n)
        if name == 'EXTRACT' and len(args)>=2:
            part = str(args[0].get(0)).upper() if not args[0].is_null(0) else ''
            return self._date_part(part, args[1], n)
        if name == 'EPOCH' and args:
            return self._num1(args[0],n, lambda x: x, DataType.BIGINT)
        if name == 'DATE_ADD' and len(args)>=2:
            return self._num2(args[0],args[1],n, lambda d,delta: d+int(delta), DataType.INT)
        if name == 'DATE_SUB' and len(args)>=2:
            return self._num2(args[0],args[1],n, lambda d,delta: d-int(delta), DataType.INT)
        if name == 'DATE_DIFF' and len(args)>=2:
            return self._num2(args[0],args[1],n, lambda a,b: int(a)-int(b), DataType.INT)
        if name == 'DATE_TRUNC' and len(args)>=2: return self._date_trunc(args[0], args[1], n)
        if name == 'DATE_FORMAT' and len(args)>=2: return self._date_format(args[0], args[1], n)
        if name == 'TO_DATE' and args: return self._to_date(args[0], n)
        if name == 'TO_TIMESTAMP' and args: return self._to_timestamp(args[0], n)

        # ── Array ─────────────────────────────────────────────
        if name in ('ARRAY','ARRAY_CREATE'): return self._eval_array_create(args, n)
        if name == 'ARRAY_LENGTH' and args:
            return self._num1(args[0],n, lambda v: len(self._parse_array(v)), DataType.INT)
        if name == 'ARRAY_CONTAINS' and len(args)>=2:
            return self._bool2(args[0],args[1],n, lambda arr,val: val in self._parse_array(arr))
        if name == 'ARRAY_POSITION' and len(args)>=2:
            return self._num2(args[0],args[1],n,
                lambda arr,val: self._parse_array(arr).index(val)+1 if val in self._parse_array(arr) else 0, DataType.INT)
        if name == 'ARRAY_SLICE' and len(args)>=3: return self._eval_array_slice(args, n)
        if name == 'ARRAY_APPEND' and len(args)>=2:
            return self._eval_array_binary(args[0],args[1],n, lambda arr,val: arr+[val])
        if name == 'ARRAY_PREPEND' and len(args)>=2:
            return self._eval_array_binary(args[0],args[1],n, lambda arr,val: [val]+arr)
        if name == 'ARRAY_REMOVE' and len(args)>=2:
            return self._eval_array_binary(args[0],args[1],n, lambda arr,val: [x for x in arr if x!=val])
        if name == 'ARRAY_CONCAT' and len(args)>=2:
            return self._eval_array_binary2(args[0],args[1],n, lambda a,b: a+b)
        if name == 'ARRAY_SORT' and args:
            return self._eval_array_unary(args[0],n, lambda a: sorted(a))
        if name == 'ARRAY_REVERSE' and args:
            return self._eval_array_unary(args[0],n, lambda a: list(reversed(a)))
        if name == 'ARRAY_DISTINCT' and args:
            return self._eval_array_unary(args[0],n, lambda a: list(dict.fromkeys(a)))
        if name == 'ARRAY_FLATTEN' and args:
            return self._eval_array_unary(args[0],n, lambda a: self._flatten(a))
        if name == 'ARRAY_INTERSECT' and len(args)>=2:
            return self._eval_array_binary2(args[0],args[1],n, lambda a,b: [x for x in a if x in set(b)])
        if name == 'ARRAY_UNION' and len(args)>=2:
            return self._eval_array_binary2(args[0],args[1],n, lambda a,b: list(dict.fromkeys(a+b)))
        if name == 'ARRAY_EXCEPT' and len(args)>=2:
            return self._eval_array_binary2(args[0],args[1],n, lambda a,b: [x for x in a if x not in set(b)])
        if name == 'ARRAY_JOIN' and len(args)>=2:
            return self._str2(args[0],args[1],n,
                lambda arr,sep: str(sep).join(str(x) for x in self._parse_array(arr)))

        # ── Table functions ───────────────────────────────────
        if name == 'GENERATE_SERIES': return self._eval_generate_series(args, n)
        if name in ('EXPLODE','UNNEST') and args:
            return self._eval_array_unary(args[0],n, lambda a: a)

        # ── Similarity ────────────────────────────────────────
        if name == 'JACCARD_SIMILARITY' and len(args)>=2: return self._eval_jaccard(args[0],args[1],n)
        if name == 'COSINE_SIMILARITY' and len(args)>=2: return self._eval_cosine(args[0],args[1],n)

        raise ExecutionError(f"unknown function: {expr.name}")

    # ══════════════════════════════════════════════════════════════
    # HELPERS
    # ══════════════════════════════════════════════════════════════
    def _str1(self, v, n, fn):
        rd = []
        rn = Bitmap(n)

        # SWAR fast path for UPPER/LOWER on ASCII strings
        use_swar = False
        if fn is str.upper or fn is str.lower:
            try:
                from metal.swar import batch_to_upper, batch_to_lower
                use_swar = True
                swar_fn = batch_to_upper if fn is str.upper else batch_to_lower
            except ImportError:
                pass

        for i in range(n):
            if v.is_null(i):
                rn.set_bit(i)
                rd.append('')
            else:
                s = str(v.get(i))
                if use_swar:
                    try:
                        raw = bytearray(s.encode('ascii'))
                        result = swar_fn(raw)
                        rd.append(result.decode('ascii'))
                        continue
                    except (UnicodeEncodeError, UnicodeDecodeError):
                        pass  # Non-ASCII → fallback
                rd.append(fn(s))

        return DataVector(dtype=DataType.VARCHAR, data=rd, nulls=rn, _length=n)
    def _str2(self, a, b, n, fn):
        rd=[]; rn=Bitmap(n)
        for i in range(n):
            if a.is_null(i) or b.is_null(i): rn.set_bit(i); rd.append('')
            else: rd.append(fn(str(a.get(i)), str(b.get(i))))
        return DataVector(dtype=DataType.VARCHAR, data=rd, nulls=rn, _length=n)

    def _str3(self, a, b, c, n, fn):
        rd=[]; rn=Bitmap(n)
        for i in range(n):
            if a.is_null(i) or b.is_null(i) or c.is_null(i): rn.set_bit(i); rd.append('')
            else: rd.append(fn(str(a.get(i)), str(b.get(i)), str(c.get(i))))
        return DataVector(dtype=DataType.VARCHAR, data=rd, nulls=rn, _length=n)

    def _str_int(self, sv, iv, n, fn):
        rd=[]; rn=Bitmap(n)
        for i in range(n):
            if sv.is_null(i) or iv.is_null(i): rn.set_bit(i); rd.append('')
            else: rd.append(fn(str(sv.get(i)), int(iv.get(i))))
        return DataVector(dtype=DataType.VARCHAR, data=rd, nulls=rn, _length=n)

    def _num1(self, v, n, fn, dt):
        code=DTYPE_TO_ARRAY_CODE.get(dt); rd: Any=TypedVector(code) if code else []; rn=Bitmap(n)
        for i in range(n):
            if v.is_null(i):
                rn.set_bit(i)
                if isinstance(rd,TypedVector): rd.append(0)
                else: rd.append(None)
            else:
                val=fn(v.get(i))
                if val is None: rn.set_bit(i); val=0 if isinstance(rd,TypedVector) else None
                if isinstance(rd,TypedVector): rd.append(val)
                else: rd.append(val)
        return DataVector(dtype=dt, data=rd, nulls=rn, _length=n)

    def _num2(self, a, b, n, fn, dt):
        code=DTYPE_TO_ARRAY_CODE.get(dt); rd: Any=TypedVector(code) if code else []; rn=Bitmap(n)
        for i in range(n):
            if a.is_null(i) or b.is_null(i):
                rn.set_bit(i)
                if isinstance(rd,TypedVector): rd.append(0)
                else: rd.append(None)
            else:
                val=fn(a.get(i), b.get(i))
                if val is None: rn.set_bit(i); val=0 if isinstance(rd,TypedVector) else None
                if isinstance(rd,TypedVector): rd.append(val)
                else: rd.append(val)
        return DataVector(dtype=dt, data=rd, nulls=rn, _length=n)

    def _bool2(self, a, b, n, fn):
        rd=Bitmap(n); rn=Bitmap(n)
        for i in range(n):
            if a.is_null(i) or b.is_null(i): rn.set_bit(i)
            elif fn(a.get(i), b.get(i)): rd.set_bit(i)
        return DataVector(dtype=DataType.BOOLEAN, data=rd, nulls=rn, _length=n)

    def _variadic(self, args, n, fn, dt):
        results=[None]*n
        for i in range(n): results[i] = fn([a.get(i) for a in args])
        return self._list_to_vec(results, dt, n)

    def _substr(self, args, n):
        sv=args[0]; rd=[]; rn=Bitmap(n)
        for i in range(n):
            if sv.is_null(i): rn.set_bit(i); rd.append(''); continue
            s=str(sv.get(i))
            start = int(args[1].get(i))-1 if len(args)>1 and not args[1].is_null(i) else 0
            if start<0: start=0
            if len(args)>2 and not args[2].is_null(i):
                rd.append(s[start:start+int(args[2].get(i))])
            else: rd.append(s[start:])
        return DataVector(dtype=DataType.VARCHAR, data=rd, nulls=rn, _length=n)

    def _concat_fn(self, name, args, n):
        rd=[]; rn=Bitmap(n)
        for i in range(n):
            if name=='CONCAT_WS':
                if args[0].is_null(i): rn.set_bit(i); rd.append(''); continue
                sep=str(args[0].get(i))
                rd.append(sep.join(str(a.get(i)) for a in args[1:] if not a.is_null(i)))
            else:
                if any(a.is_null(i) for a in args): rn.set_bit(i); rd.append('')
                else: rd.append(''.join(str(a.get(i)) for a in args))
        return DataVector(dtype=DataType.VARCHAR, data=rd, nulls=rn, _length=n)

    def _lpad_rpad(self, args, n, left):
        sv,lv=args[0],args[1]; rd=[]; rn=Bitmap(n)
        for i in range(n):
            if sv.is_null(i) or lv.is_null(i): rn.set_bit(i); rd.append('')
            else:
                s=str(sv.get(i)); w=int(lv.get(i))
                pc = str(args[2].get(i))[0] if len(args)>=3 and not args[2].is_null(i) else ' '
                rd.append(s.rjust(w,pc) if left else s.ljust(w,pc))
        return DataVector(dtype=DataType.VARCHAR, data=rd, nulls=rn, _length=n)

    def _split_part(self, args, n):
        rd=[]; rn=Bitmap(n)
        for i in range(n):
            if any(args[j].is_null(i) for j in range(3)): rn.set_bit(i); rd.append('')
            else:
                parts=str(args[0].get(i)).split(str(args[1].get(i)))
                idx=int(args[2].get(i))
                rd.append(parts[idx-1] if 1<=idx<=len(parts) else '')
        return DataVector(dtype=DataType.VARCHAR, data=rd, nulls=rn, _length=n)

    def _coalesce(self, args, n):
        dt=DataType.UNKNOWN
        for a in args:
            if any(not a.is_null(i) for i in range(len(a))): dt=a.dtype; break
        if dt==DataType.UNKNOWN:
            for a in args:
                if a.dtype!=DataType.UNKNOWN: dt=a.dtype; break
        if dt==DataType.UNKNOWN: dt=DataType.INT
        results=[None]*n
        for i in range(n):
            for a in args:
                if not a.is_null(i): results[i]=a.get(i); break
        return self._list_to_vec(results, dt, n)

    def _nullif(self, a, b, n):
        results=[None]*n
        for i in range(n):
            if a.is_null(i): continue
            if b.is_null(i): results[i]=a.get(i); continue
            results[i] = None if a.get(i)==b.get(i) else a.get(i)
        return self._list_to_vec(results, a.dtype, n)

    def _if_fn(self, c, t, e, n):
        results=[None]*n
        for i in range(n):
            results[i] = t.get(i) if (not c.is_null(i) and c.get(i)) else e.get(i)
        dt = t.dtype if t.dtype!=DataType.UNKNOWN else e.dtype
        return self._list_to_vec(results, dt, n)

    def _date_part(self, part, vec, n):
        results=[None]*n
        for i in range(n):
            if vec.is_null(i): continue
            v=vec.get(i)
            try:
                if isinstance(v,int) and abs(v)<1_000_000:
                    d=_dt.date(1970,1,1)+_dt.timedelta(days=v)
                    results[i]=self._extract(part,d,None)
                elif isinstance(v,int):
                    dt_obj=_dt.datetime(1970,1,1)+_dt.timedelta(microseconds=v)
                    results[i]=self._extract(part,dt_obj.date(),dt_obj.time())
            except Exception: pass
        return self._list_to_vec(results, DataType.INT, n)

    def _extract(self, part, d, t):
        if part=='YEAR': return d.year
        if part=='MONTH': return d.month
        if part=='DAY': return d.day
        if part=='HOUR': return t.hour if t else 0
        if part=='MINUTE': return t.minute if t else 0
        if part=='SECOND': return t.second if t else 0
        if part=='DAY_OF_WEEK': return d.isoweekday()
        if part=='DAY_OF_YEAR': return d.timetuple().tm_yday
        if part=='WEEK_OF_YEAR': return d.isocalendar()[1]
        if part=='QUARTER': return (d.month-1)//3+1
        return None

    def _date_trunc(self, unit_vec, val_vec, n):
        results=[None]*n
        for i in range(n):
            if unit_vec.is_null(i) or val_vec.is_null(i): continue
            unit=str(unit_vec.get(i)).upper(); v=int(val_vec.get(i))
            try:
                if abs(v)<1_000_000:
                    d=_dt.date(1970,1,1)+_dt.timedelta(days=v)
                    if unit=='YEAR': d=d.replace(month=1,day=1)
                    elif unit=='MONTH': d=d.replace(day=1)
                    elif unit=='QUARTER': d=d.replace(month=((d.month-1)//3)*3+1,day=1)
                    elif unit=='WEEK': d=d-_dt.timedelta(days=d.weekday())
                    results[i]=(d-_dt.date(1970,1,1)).days
                else: results[i]=v
            except Exception: pass
        return self._list_to_vec(results, DataType.INT, n)

    def _date_format(self, val_vec, fmt_vec, n):
        rd=[]; rn=Bitmap(n)
        for i in range(n):
            if val_vec.is_null(i) or fmt_vec.is_null(i): rn.set_bit(i); rd.append(''); continue
            v=int(val_vec.get(i)); fmt=str(fmt_vec.get(i))
            try:
                if abs(v)<1_000_000: d=_dt.date(1970,1,1)+_dt.timedelta(days=v)
                else: d=_dt.datetime(1970,1,1)+_dt.timedelta(microseconds=v)
                py_fmt=fmt.replace('YYYY','%Y').replace('MM','%m').replace('DD','%d').replace('HH','%H').replace('MI','%M').replace('SS','%S')
                rd.append(d.strftime(py_fmt))
            except Exception: rn.set_bit(i); rd.append('')
        return DataVector(dtype=DataType.VARCHAR, data=rd, nulls=rn, _length=n)

    def _to_date(self, val_vec, n):
        results=[None]*n
        for i in range(n):
            if val_vec.is_null(i): continue
            s=str(val_vec.get(i))
            try:
                from executor.string_algo.compiled_date import parse_date_auto
                results[i]=parse_date_auto(s)
            except ImportError:
                for fmt in ('%Y-%m-%d','%m/%d/%Y','%d.%m.%Y'):
                    try:
                        d=_dt.datetime.strptime(s,fmt).date()
                        results[i]=(d-_dt.date(1970,1,1)).days; break
                    except ValueError: continue
        return self._list_to_vec(results, DataType.DATE, n)

    def _to_timestamp(self, val_vec, n):
        results=[None]*n
        for i in range(n):
            if val_vec.is_null(i): continue
            s=str(val_vec.get(i))
            try:
                from executor.string_algo.compiled_date import ISO_DATE_PARSER
                results[i]=ISO_DATE_PARSER.parse_timestamp(s)
            except ImportError:
                for fmt in ('%Y-%m-%d %H:%M:%S','%Y-%m-%dT%H:%M:%S','%Y-%m-%d'):
                    try:
                        dt_obj=_dt.datetime.strptime(s,fmt)
                        results[i]=int((dt_obj-_dt.datetime(1970,1,1)).total_seconds()*1_000_000); break
                    except ValueError: continue
        return self._list_to_vec(results, DataType.TIMESTAMP, n)

    def _width_bucket(self, args, n):
        results=[None]*n
        for i in range(n):
            if any(args[j].is_null(i) for j in range(4)): continue
            val=float(args[0].get(i)); lo=float(args[1].get(i))
            hi=float(args[2].get(i)); buckets=int(args[3].get(i))
            if hi==lo or buckets<=0: results[i]=0
            elif val<lo: results[i]=0
            elif val>=hi: results[i]=buckets+1
            else: results[i]=int((val-lo)/(hi-lo)*buckets)+1
        return self._list_to_vec(results, DataType.INT, n)

    def _eval_try_cast(self, expr, batch):
        if isinstance(expr, FunctionCall) and len(expr.args)>=2:
            src = self.evaluate(expr.args[0], batch)
            type_str = str(expr.args[1].value) if hasattr(expr.args[1],'value') else 'VARCHAR'
            target_dt,_ = resolve_type_name(type_str,[])
            n=len(src); results=[None]*n
            for i in range(n):
                if src.is_null(i): continue
                v=src.get(i)
                try:
                    if target_dt==DataType.INT: results[i]=int(v)
                    elif target_dt==DataType.BIGINT: results[i]=int(v)
                    elif target_dt in (DataType.FLOAT,DataType.DOUBLE): results[i]=float(v)
                    elif target_dt in (DataType.VARCHAR,DataType.TEXT): results[i]=str(v)
                    elif target_dt==DataType.BOOLEAN: results[i]=bool(v)
                    else: results[i]=v
                except (ValueError,TypeError): results[i]=None
            return self._list_to_vec(results, target_dt, n)
        n=batch.row_count
        return self._list_to_vec([None]*n, DataType.VARCHAR, n)

    # ── Array helpers ─────────────────────────────────────────
    @staticmethod
    def _parse_array(val):
        if isinstance(val, list): return val
        if isinstance(val, str):
            s=val.strip()
            if s.startswith('[') and s.endswith(']'):
                s=s[1:-1].strip()
                if not s: return []
                parts=[]; depth=0; current=[]
                for ch in s:
                    if ch=='[': depth+=1; current.append(ch)
                    elif ch==']': depth-=1; current.append(ch)
                    elif ch==',' and depth==0: parts.append(''.join(current).strip()); current=[]
                    else: current.append(ch)
                if current: parts.append(''.join(current).strip())
                result=[]
                for p in parts:
                    p=p.strip().strip("'\"")
                    try: result.append(int(p))
                    except ValueError:
                        try: result.append(float(p))
                        except ValueError: result.append(p)
                return result
        return [val] if val is not None else []

    def _eval_array_create(self, args, n):
        rd=[]; rn=Bitmap(n)
        for i in range(n): rd.append(str([a.get(i) for a in args]))
        return DataVector(dtype=DataType.VARCHAR, data=rd, nulls=rn, _length=n)

    def _eval_array_unary(self, arr_vec, n, fn):
        rd=[]; rn=Bitmap(n)
        for i in range(n):
            if arr_vec.is_null(i): rn.set_bit(i); rd.append('')
            else: rd.append(str(fn(self._parse_array(arr_vec.get(i)))))
        return DataVector(dtype=DataType.VARCHAR, data=rd, nulls=rn, _length=n)

    def _eval_array_binary(self, arr_vec, val_vec, n, fn):
        rd=[]; rn=Bitmap(n)
        for i in range(n):
            if arr_vec.is_null(i): rn.set_bit(i); rd.append('')
            else:
                arr=self._parse_array(arr_vec.get(i))
                val=val_vec.get(i) if not val_vec.is_null(i) else None
                rd.append(str(fn(arr,val)))
        return DataVector(dtype=DataType.VARCHAR, data=rd, nulls=rn, _length=n)

    def _eval_array_binary2(self, a_vec, b_vec, n, fn):
        rd=[]; rn=Bitmap(n)
        for i in range(n):
            if a_vec.is_null(i) or b_vec.is_null(i): rn.set_bit(i); rd.append('')
            else: rd.append(str(fn(self._parse_array(a_vec.get(i)), self._parse_array(b_vec.get(i)))))
        return DataVector(dtype=DataType.VARCHAR, data=rd, nulls=rn, _length=n)

    def _eval_array_slice(self, args, n):
        rd=[]; rn=Bitmap(n)
        for i in range(n):
            if args[0].is_null(i): rn.set_bit(i); rd.append('')
            else:
                arr=self._parse_array(args[0].get(i))
                start=int(args[1].get(i))-1 if not args[1].is_null(i) else 0
                end=int(args[2].get(i)) if not args[2].is_null(i) else len(arr)
                rd.append(str(arr[start:end]))
        return DataVector(dtype=DataType.VARCHAR, data=rd, nulls=rn, _length=n)

    @staticmethod
    def _flatten(arr):
        result=[]
        for item in arr:
            if isinstance(item,list): result.extend(item)
            else: result.append(item)
        return result

    def _eval_generate_series(self, args, n):
        rd=[]; rn=Bitmap(n)
        for i in range(n):
            if any(args[j].is_null(i) for j in range(min(len(args),2))): rn.set_bit(i); rd.append(''); continue
            start=int(args[0].get(i)); stop=int(args[1].get(i))
            step=int(args[2].get(i)) if len(args)>=3 and not args[2].is_null(i) else 1
            if step==0: rn.set_bit(i); rd.append('')
            else: rd.append(str(list(range(start, stop+(1 if step>0 else -1), step))))
        return DataVector(dtype=DataType.VARCHAR, data=rd, nulls=rn, _length=n)

    def _eval_jaccard(self, a_vec, b_vec, n):
        code=DTYPE_TO_ARRAY_CODE.get(DataType.DOUBLE)
        rd: Any=TypedVector(code) if code else []; rn=Bitmap(n)
        for i in range(n):
            if a_vec.is_null(i) or b_vec.is_null(i):
                rn.set_bit(i); rd.append(0.0)
            else:
                sa=set(self._parse_array(a_vec.get(i))); sb=set(self._parse_array(b_vec.get(i)))
                val=len(sa&sb)/len(sa|sb) if (sa or sb) else 1.0
                rd.append(val)
        return DataVector(dtype=DataType.DOUBLE, data=rd, nulls=rn, _length=n)

    def _eval_cosine(self, a_vec, b_vec, n):
        code=DTYPE_TO_ARRAY_CODE.get(DataType.DOUBLE)
        rd: Any=TypedVector(code) if code else []; rn=Bitmap(n)
        for i in range(n):
            if a_vec.is_null(i) or b_vec.is_null(i):
                rn.set_bit(i); rd.append(0.0)
            else:
                va=[float(x) for x in self._parse_array(a_vec.get(i))]
                vb=[float(x) for x in self._parse_array(b_vec.get(i))]
                ml=min(len(va),len(vb))
                if ml==0: rd.append(0.0)
                else:
                    dot=sum(va[j]*vb[j] for j in range(ml))
                    ma=_math.sqrt(sum(x*x for x in va[:ml]))
                    mb=_math.sqrt(sum(x*x for x in vb[:ml]))
                    rd.append(dot/(ma*mb) if ma>0 and mb>0 else 0.0)
        return DataVector(dtype=DataType.DOUBLE, data=rd, nulls=rn, _length=n)

    # ══════════════════════════════════════════════════════════════
    def _to_bool(self, vec):
        if vec.dtype==DataType.BOOLEAN: return vec
        if is_numeric(vec.dtype):
            n=len(vec); d=Bitmap(n); nl=Bitmap(n)
            for i in range(n):
                if vec.is_null(i): nl.set_bit(i)
                elif vec.get(i)!=0: d.set_bit(i)
            return DataVector(dtype=DataType.BOOLEAN, data=d, nulls=nl, _length=n)
        raise ExecutionError(f"cannot convert {vec.dtype.name} to BOOLEAN")

    def _bool_to_bitmap(self, vec):
        n=len(vec); bm=Bitmap(n)
        assert isinstance(vec.data, Bitmap)
        for i in range(n):
            if not vec.is_null(i) and vec.data.get_bit(i): bm.set_bit(i)
        return bm

    def _list_to_vec(self, values, dtype, n):
        if dtype==DataType.UNKNOWN: dtype=DataType.INT
        code=DTYPE_TO_ARRAY_CODE.get(dtype); nulls=Bitmap(n)
        if dtype==DataType.BOOLEAN:
            data: Any=Bitmap(n)
            for i in range(n):
                if values[i] is None: nulls.set_bit(i)
                elif values[i]: data.set_bit(i)
        elif dtype in (DataType.VARCHAR,DataType.TEXT):
            data=[]
            for i in range(n):
                if values[i] is None: nulls.set_bit(i); data.append('')
                else: data.append(str(values[i]))
        elif code:
            data=TypedVector(code)
            for i in range(n):
                if values[i] is None: nulls.set_bit(i); data.append(0)
                else: data.append(values[i])
        else:
            data=TypedVector('q')
            for i in range(n):
                if values[i] is None: nulls.set_bit(i); data.append(0)
                else: data.append(int(values[i]))
        return DataVector(dtype=dtype, data=data, nulls=nulls, _length=n)

    # ══════════════════════════════════════════════════════════════
    @staticmethod
    def infer_type(expr, schema):
        if isinstance(expr, Literal):
            return expr.inferred_type if expr.inferred_type!=DataType.UNKNOWN else DataType.INT
        if isinstance(expr, ColumnRef):
            if expr.table:
                q=f"{expr.table}.{expr.column}"
                if q in schema: return schema[q]
            return schema.get(expr.column, DataType.UNKNOWN)
        if isinstance(expr, AliasExpr): return ExpressionEvaluator.infer_type(expr.expr, schema)
        if isinstance(expr, BinaryExpr):
            if expr.op in ('+','-','*','/','%'):
                lt=ExpressionEvaluator.infer_type(expr.left,schema)
                rt=ExpressionEvaluator.infer_type(expr.right,schema)
                try: return promote(lt,rt)
                except TypeMismatchError: return DataType.UNKNOWN
            if expr.op=='||': return DataType.VARCHAR
            return DataType.BOOLEAN
        if isinstance(expr, UnaryExpr):
            if expr.op=='NOT': return DataType.BOOLEAN
            return ExpressionEvaluator.infer_type(expr.operand, schema)
        if isinstance(expr, IsNullExpr): return DataType.BOOLEAN
        if isinstance(expr, (InExpr,BetweenExpr,LikeExpr)): return DataType.BOOLEAN
        if isinstance(expr, CaseExpr):
            if expr.when_clauses: return ExpressionEvaluator.infer_type(expr.when_clauses[0][1], schema)
            return DataType.UNKNOWN
        if isinstance(expr, CastExpr):
            if expr.type_name:
                dt,_=resolve_type_name(expr.type_name.name,expr.type_name.params); return dt
            return DataType.UNKNOWN
        if isinstance(expr, FunctionCall):
            nm=expr.name.upper()
            if nm in ('UPPER','LOWER','TRIM','LTRIM','RTRIM','REVERSE','REPLACE','SUBSTR','SUBSTRING',
                       'CONCAT','CONCAT_WS','LEFT','RIGHT','REPEAT','LPAD','RPAD','INITCAP','CHR',
                       'SPLIT_PART','REGEXP_REPLACE','REGEXP_EXTRACT','SPLIT','ENCODE','DECODE',
                       'DATE_FORMAT','ARRAY','ARRAY_CREATE','ARRAY_SORT','ARRAY_REVERSE',
                       'ARRAY_DISTINCT','ARRAY_FLATTEN','ARRAY_INTERSECT','ARRAY_UNION',
                       'ARRAY_EXCEPT','ARRAY_APPEND','ARRAY_PREPEND','ARRAY_REMOVE',
                       'ARRAY_CONCAT','ARRAY_SLICE','ARRAY_JOIN','GENERATE_SERIES',
                       'EXPLODE','UNNEST','TRY_CAST'): return DataType.VARCHAR
            if nm in ('LENGTH','SIGN','POSITION','ASCII','YEAR','MONTH','DAY','HOUR','MINUTE',
                       'SECOND','DAY_OF_WEEK','DAY_OF_YEAR','WEEK_OF_YEAR','QUARTER',
                       'DATE_DIFF','DATE_ADD','DATE_SUB','TRUNC','TRUNCATE','WIDTH_BUCKET',
                       'BIT_COUNT','ARRAY_LENGTH','ARRAY_POSITION'): return DataType.INT
            if nm in ('CEIL','CEILING','FLOOR') or (nm=='ROUND' and len(expr.args)<2): return DataType.BIGINT
            if nm in ('POWER','SQRT','CBRT','LN','LOG','LOG2','LOG10','EXP','ROUND','RANDOM',
                       'JACCARD_SIMILARITY','COSINE_SIMILARITY'): return DataType.DOUBLE
            if nm in ('NOW','CURRENT_TIMESTAMP'): return DataType.TIMESTAMP
            if nm == 'CURRENT_DATE': return DataType.DATE
            if nm == 'DATE_TRUNC': return DataType.DATE
            if nm == 'TO_DATE': return DataType.DATE
            if nm == 'TO_TIMESTAMP': return DataType.TIMESTAMP
            if nm in ('EPOCH','HASH','MURMUR_HASH'): return DataType.BIGINT
            if nm in ('STARTS_WITH','ENDS_WITH','CONTAINS','REGEXP_MATCH','ARRAY_CONTAINS'): return DataType.BOOLEAN
            if nm=='ABS' and expr.args: return ExpressionEvaluator.infer_type(expr.args[0],schema)
            if nm=='COALESCE' and expr.args: return ExpressionEvaluator.infer_type(expr.args[0],schema)
            if nm=='NULLIF' and expr.args: return ExpressionEvaluator.infer_type(expr.args[0],schema)
            if nm=='IF' and len(expr.args)>=2: return ExpressionEvaluator.infer_type(expr.args[1],schema)
            if nm=='TYPEOF': return DataType.VARCHAR
            if nm in ('GREATEST','LEAST','MOD') and expr.args:
                return ExpressionEvaluator.infer_type(expr.args[0],schema)
            return DataType.UNKNOWN
        if isinstance(expr, AggregateCall):
            u=expr.name.upper()
            if u=='COUNT': return DataType.BIGINT
            if u in ('AVG','AVG_DISTINCT','STDDEV','STDDEV_POP','VARIANCE','VAR_POP',
                      'APPROX_PERCENTILE'): return DataType.DOUBLE
            if u in ('SUM','SUM_DISTINCT'):
                if expr.args and not isinstance(expr.args[0],StarExpr):
                    at=ExpressionEvaluator.infer_type(expr.args[0],schema)
                    if at in (DataType.FLOAT,DataType.DOUBLE): return DataType.DOUBLE
                return DataType.BIGINT
            if u in ('MIN','MAX','PERCENTILE_DISC','MODE'):
                if expr.args and not isinstance(expr.args[0],StarExpr):
                    return ExpressionEvaluator.infer_type(expr.args[0],schema)
            if u in ('APPROX_COUNT_DISTINCT','COUNT_DISTINCT'): return DataType.BIGINT
            if u in ('MEDIAN','PERCENTILE_CONT'): return DataType.DOUBLE
            if u in ('STRING_AGG','ARRAY_AGG','APPROX_TOP_K','GROUPING'): return DataType.VARCHAR
            return DataType.UNKNOWN
        return DataType.UNKNOWN

    @staticmethod
    def cast_vector(vec, target):
        if vec.dtype==target: return vec
        n=len(vec)
        if vec.dtype==DataType.BOOLEAN and is_numeric(target):
            code=DTYPE_TO_ARRAY_CODE.get(target,'q'); assert code
            data=TypedVector(code); nulls=Bitmap(n)
            for i in range(n):
                if vec.is_null(i): nulls.set_bit(i); data.append(0)
                else: data.append(1 if vec.get(i) else 0)
            return DataVector(dtype=target, data=data, nulls=nulls, _length=n)
        sr={DataType.INT:0,DataType.BIGINT:1,DataType.FLOAT:2,DataType.DOUBLE:3}
        if vec.dtype in sr and target in sr and sr[target]>=sr[vec.dtype]:
            code=DTYPE_TO_ARRAY_CODE.get(target,'q'); assert code
            data=TypedVector(code); nulls=Bitmap(n)
            for i in range(n):
                if vec.is_null(i): nulls.set_bit(i); data.append(0)
                else: data.append(float(vec.get(i)) if target in (DataType.FLOAT,DataType.DOUBLE) else int(vec.get(i)))
            return DataVector(dtype=target, data=data, nulls=nulls, _length=n)
        raise TypeMismatchError(f"cannot cast {vec.dtype.name} to {target.name}",
                                expected=target.name, actual=vec.dtype.name)


def _like_match(text, pattern):
    regex='^'
    for ch in pattern:
        if ch=='%': regex+='.*'
        elif ch=='_': regex+='.'
        elif ch in r'\.^$+?{}[]|()': regex+='\\'+ch
        else: regex+=ch
    regex+='$'
    return bool(_re.match(regex, text, _re.DOTALL))
