from __future__ import annotations
"""Expression evaluator — vectorised evaluation over DataVectors."""
import operator as _op
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


class ExpressionEvaluator:
    def __init__(self, registry: Optional[Any] = None) -> None:
        self._registry = registry

    # ==================================================================
    def evaluate(self, expr: Any, batch: VectorBatch) -> DataVector:
        if isinstance(expr, Literal):
            return self._eval_literal(expr, batch.row_count)
        if isinstance(expr, ColumnRef):
            return self._eval_column_ref(expr, batch)
        if isinstance(expr, AliasExpr):
            return self.evaluate(expr.expr, batch)
        if isinstance(expr, BinaryExpr):
            l = self.evaluate(expr.left, batch)
            r = self.evaluate(expr.right, batch)
            return self._eval_binary(expr.op, l, r)
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
            raise ExecutionError("aggregate in non-aggregate context")
        if isinstance(expr, StarExpr):
            raise ExecutionError("internal: StarExpr in eval")
        raise ExecutionError(f"unknown expression: {type(expr).__name__}")

    def evaluate_predicate(self, expr: Any, batch: VectorBatch) -> Bitmap:
        vec = self.evaluate(expr, batch)
        if vec.dtype == DataType.BOOLEAN:
            return self._bool_to_bitmap(vec)
        if is_numeric(vec.dtype):
            bm = Bitmap(len(vec))
            for i in range(len(vec)):
                if not vec.is_null(i) and vec.get(i) != 0:
                    bm.set_bit(i)
            return bm
        raise ExecutionError(f"WHERE must be boolean, got {vec.dtype.name}")

    # ==================================================================
    # ColumnRef — handles qualified (table.column) and unqualified
    # ==================================================================
    def _eval_column_ref(self, expr: ColumnRef, batch: VectorBatch) -> DataVector:
        # Try qualified name first (for JOIN queries)
        if expr.table is not None:
            qualified = f"{expr.table}.{expr.column}"
            if qualified in batch.columns:
                return batch.columns[qualified]
        # Try unqualified
        if expr.column in batch.columns:
            return batch.columns[expr.column]
        # Try scanning all columns for suffix match (e.g. column 'u.name' matches ref 'name')
        if expr.table is not None:
            qualified = f"{expr.table}.{expr.column}"
            raise ExecutionError(f"column '{qualified}' not found")
        raise ExecutionError(f"column '{expr.column}' not found")

    # ==================================================================
    # Literal
    # ==================================================================
    def _eval_literal(self, lit: Literal, n: int) -> DataVector:
        if lit.value is None:
            dt = lit.inferred_type if lit.inferred_type != DataType.UNKNOWN else DataType.INT
            nulls = Bitmap(n)
            for i in range(n):
                nulls.set_bit(i)
            return DataVector.from_nulls(dt, n, nulls)
        dtype = lit.inferred_type
        if dtype == DataType.UNKNOWN:
            dtype = DataType.INT
        code = DTYPE_TO_ARRAY_CODE.get(dtype)
        if code is not None:
            data: Any = TypedVector(code, [lit.value] * n)
        elif dtype in (DataType.VARCHAR, DataType.TEXT):
            data = [lit.value] * n
        elif dtype == DataType.BOOLEAN:
            data = Bitmap(n)
            if lit.value:
                for i in range(n):
                    data.set_bit(i)
        else:
            data = TypedVector('q', [lit.value] * n)
        return DataVector(dtype=dtype, data=data, nulls=Bitmap(n), _length=n)

    # ==================================================================
    # Binary
    # ==================================================================
    def _eval_binary(self, op: str, left: DataVector, right: DataVector) -> DataVector:
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
        raise ExecutionError(f"unsupported op: {op}")

    def _eval_arith(self, op: str, left: DataVector, right: DataVector) -> DataVector:
        target = promote(left.dtype, right.dtype)
        lv = self.cast_vector(left, target)
        rv = self.cast_vector(right, target)
        n = len(lv)
        is_int = target in (DataType.INT, DataType.BIGINT, DataType.BOOLEAN)
        ops = {'+': _op.add, '-': _op.sub, '*': _op.mul}
        code = DTYPE_TO_ARRAY_CODE.get(target)
        rd: Any = TypedVector(code) if code else []
        rn = Bitmap(n)
        for i in range(n):
            if lv.is_null(i) or rv.is_null(i):
                rn.set_bit(i)
                if isinstance(rd, TypedVector):
                    rd.append(0)
                else:
                    rd.append(None)
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
                    val = int(a / b) if is_int else a / b
            elif op == '%':
                if b == 0:
                    raise DivisionByZeroError()
                val = a % b
            else:
                raise ExecutionError(f"unknown arith: {op}")
            if target == DataType.INT and not (-2_147_483_648 <= val <= 2_147_483_647):
                raise NumericOverflowError(f"overflow: {val}")
            if isinstance(rd, TypedVector):
                rd.append(val)
            else:
                rd.append(val)
        return DataVector(dtype=target, data=rd, nulls=rn, _length=n)

    def _eval_cmp(self, op: str, left: DataVector, right: DataVector) -> DataVector:
        if is_numeric(left.dtype) and is_numeric(right.dtype):
            t = promote(left.dtype, right.dtype)
            lv = self.cast_vector(left, t)
            rv = self.cast_vector(right, t)
        elif is_string(left.dtype) and is_string(right.dtype):
            lv, rv = left, right
        elif left.dtype == DataType.BOOLEAN and right.dtype == DataType.BOOLEAN:
            lv, rv = left, right
        else:
            try:
                t = promote(left.dtype, right.dtype)
                lv = self.cast_vector(left, t)
                rv = self.cast_vector(right, t)
            except TypeMismatchError:
                lv, rv = left, right
        n = len(lv)
        fns = {'=': _op.eq, '!=': _op.ne, '<': _op.lt,
               '>': _op.gt, '<=': _op.le, '>=': _op.ge}
        fn = fns[op]
        rd = Bitmap(n)
        rn = Bitmap(n)
        for i in range(n):
            if lv.is_null(i) or rv.is_null(i):
                rn.set_bit(i)
                continue
            if fn(lv.get(i), rv.get(i)):
                rd.set_bit(i)
        return DataVector(dtype=DataType.BOOLEAN, data=rd, nulls=rn, _length=n)

    def _eval_concat(self, left: DataVector, right: DataVector) -> DataVector:
        n = len(left)
        rd: list = []
        rn = Bitmap(n)
        for i in range(n):
            if left.is_null(i) or right.is_null(i):
                rn.set_bit(i)
                rd.append('')
            else:
                rd.append(str(left.get(i)) + str(right.get(i)))
        return DataVector(dtype=DataType.VARCHAR, data=rd, nulls=rn, _length=n)

    def _eval_and(self, l: DataVector, r: DataVector) -> DataVector:
        n = len(l)
        rd = Bitmap(n)
        rn = Bitmap(n)
        for i in range(n):
            ln, rnn = l.is_null(i), r.is_null(i)
            lv = l.get(i) if not ln else None
            rv = r.get(i) if not rnn else None
            if (not ln and not lv) or (not rnn and not rv):
                pass
            elif ln or rnn:
                rn.set_bit(i)
            elif lv and rv:
                rd.set_bit(i)
        return DataVector(dtype=DataType.BOOLEAN, data=rd, nulls=rn, _length=n)

    def _eval_or(self, l: DataVector, r: DataVector) -> DataVector:
        n = len(l)
        rd = Bitmap(n)
        rn = Bitmap(n)
        for i in range(n):
            ln, rnn = l.is_null(i), r.is_null(i)
            lv = l.get(i) if not ln else None
            rv = r.get(i) if not rnn else None
            if (not ln and lv) or (not rnn and rv):
                rd.set_bit(i)
            elif ln or rnn:
                rn.set_bit(i)
        return DataVector(dtype=DataType.BOOLEAN, data=rd, nulls=rn, _length=n)

    # ==================================================================
    # Unary
    # ==================================================================
    def _eval_unary(self, expr: Any, batch: VectorBatch) -> DataVector:
        op_vec = self.evaluate(expr.operand, batch)
        if expr.op == '+':
            return op_vec
        if expr.op == '-':
            n = len(op_vec)
            if not is_numeric(op_vec.dtype):
                raise ExecutionError(f"cannot negate {op_vec.dtype.name}")
            code = DTYPE_TO_ARRAY_CODE.get(op_vec.dtype)
            if code is None:
                raise ExecutionError(f"cannot negate {op_vec.dtype.name}")
            rd = TypedVector(code)
            rn = Bitmap(n)
            for i in range(n):
                if op_vec.is_null(i):
                    rn.set_bit(i)
                    rd.append(0)
                else:
                    rd.append(-op_vec.get(i))
            return DataVector(dtype=op_vec.dtype, data=rd, nulls=rn, _length=n)
        if expr.op == 'NOT':
            bv = self._to_bool(op_vec)
            n = len(bv)
            rd = Bitmap(n)
            rn = Bitmap(n)
            for i in range(n):
                if bv.is_null(i):
                    rn.set_bit(i)
                elif not bv.data.get_bit(i):
                    rd.set_bit(i)
            return DataVector(dtype=DataType.BOOLEAN, data=rd, nulls=rn, _length=n)
        raise ExecutionError(f"unsupported unary: {expr.op}")

    # ==================================================================
    # IS NULL
    # ==================================================================
    def _eval_is_null(self, expr: Any, batch: VectorBatch) -> DataVector:
        op_vec = self.evaluate(expr.expr, batch)
        n = len(op_vec)
        rd = Bitmap(n)
        for i in range(n):
            is_n = op_vec.is_null(i)
            if (is_n and not expr.negated) or (not is_n and expr.negated):
                rd.set_bit(i)
        return DataVector(dtype=DataType.BOOLEAN, data=rd, nulls=Bitmap(n), _length=n)

    # ==================================================================
    # CASE
    # ==================================================================
    def _eval_case(self, expr: CaseExpr, batch: VectorBatch) -> DataVector:
        n = batch.row_count
        results: list = [None] * n
        resolved: list = [False] * n

        if expr.operand is not None:
            op_vec = self.evaluate(expr.operand, batch)
            for cond, result in expr.when_clauses:
                cond_vec = self.evaluate(cond, batch)
                res_vec = self.evaluate(result, batch)
                for i in range(n):
                    if resolved[i]:
                        continue
                    if not op_vec.is_null(i) and not cond_vec.is_null(i) and op_vec.get(i) == cond_vec.get(i):
                        results[i] = res_vec.get(i)
                        resolved[i] = True
        else:
            for cond, result in expr.when_clauses:
                cond_vec = self.evaluate(cond, batch)
                res_vec = self.evaluate(result, batch)
                for i in range(n):
                    if resolved[i]:
                        continue
                    if not cond_vec.is_null(i) and cond_vec.get(i):
                        results[i] = res_vec.get(i)
                        resolved[i] = True

        if expr.else_expr is not None:
            else_vec = self.evaluate(expr.else_expr, batch)
            for i in range(n):
                if not resolved[i]:
                    results[i] = else_vec.get(i)

        # Determine type
        dtype = DataType.VARCHAR  # safe default for CASE
        if expr.when_clauses:
            child_schema = {cn: batch.columns[cn].dtype for cn in batch.column_names}
            dt = ExpressionEvaluator.infer_type(expr.when_clauses[0][1], child_schema)
            if dt != DataType.UNKNOWN:
                dtype = dt
        return self._list_to_vector(results, dtype, n)

    # ==================================================================
    # CAST
    # ==================================================================
    def _eval_cast(self, expr: CastExpr, batch: VectorBatch) -> DataVector:
        src = self.evaluate(expr.expr, batch)
        assert expr.type_name is not None
        target_dt, _ = resolve_type_name(expr.type_name.name, expr.type_name.params)
        if src.dtype == target_dt:
            return src
        n = len(src)
        results: list = [None] * n
        for i in range(n):
            if src.is_null(i):
                continue
            v = src.get(i)
            try:
                if target_dt == DataType.INT:
                    results[i] = int(v)
                elif target_dt == DataType.BIGINT:
                    results[i] = int(v)
                elif target_dt in (DataType.FLOAT, DataType.DOUBLE):
                    results[i] = float(v)
                elif target_dt in (DataType.VARCHAR, DataType.TEXT):
                    results[i] = str(v)
                elif target_dt == DataType.BOOLEAN:
                    results[i] = bool(v)
                else:
                    results[i] = v
            except (ValueError, TypeError):
                results[i] = None
        return self._list_to_vector(results, target_dt, n)

    # ==================================================================
    # IN
    # ==================================================================
    def _eval_in(self, expr: InExpr, batch: VectorBatch) -> DataVector:
        src = self.evaluate(expr.expr, batch)
        n = len(src)
        val_vecs = [self.evaluate(v, batch) for v in expr.values]
        rd = Bitmap(n)
        rn = Bitmap(n)
        for i in range(n):
            if src.is_null(i):
                rn.set_bit(i)
                continue
            sv = src.get(i)
            found = False
            has_null = False
            for vv in val_vecs:
                if vv.is_null(i):
                    has_null = True
                    continue
                if sv == vv.get(i):
                    found = True
                    break
            if found:
                if not expr.negated:
                    rd.set_bit(i)
            elif has_null:
                rn.set_bit(i)
            else:
                if expr.negated:
                    rd.set_bit(i)
        return DataVector(dtype=DataType.BOOLEAN, data=rd, nulls=rn, _length=n)

    # ==================================================================
    # BETWEEN
    # ==================================================================
    def _eval_between(self, expr: BetweenExpr, batch: VectorBatch) -> DataVector:
        src = self.evaluate(expr.expr, batch)
        low = self.evaluate(expr.low, batch)
        high = self.evaluate(expr.high, batch)
        n = len(src)
        rd = Bitmap(n)
        rn = Bitmap(n)
        for i in range(n):
            if src.is_null(i) or low.is_null(i) or high.is_null(i):
                rn.set_bit(i)
                continue
            sv, lv, hv = src.get(i), low.get(i), high.get(i)
            in_range = lv <= sv <= hv
            if (in_range and not expr.negated) or (not in_range and expr.negated):
                rd.set_bit(i)
        return DataVector(dtype=DataType.BOOLEAN, data=rd, nulls=rn, _length=n)

    # ==================================================================
    # LIKE
    # ==================================================================
    def _eval_like(self, expr: LikeExpr, batch: VectorBatch) -> DataVector:
        src = self.evaluate(expr.expr, batch)
        pat = self.evaluate(expr.pattern, batch)
        n = len(src)
        rd = Bitmap(n)
        rn = Bitmap(n)
        for i in range(n):
            if src.is_null(i) or pat.is_null(i):
                rn.set_bit(i)
                continue
            matched = _like_match(str(src.get(i)), str(pat.get(i)))
            if (matched and not expr.negated) or (not matched and expr.negated):
                rd.set_bit(i)
        return DataVector(dtype=DataType.BOOLEAN, data=rd, nulls=rn, _length=n)

    # ==================================================================
    # Scalar functions
    # ==================================================================
    def _eval_function(self, expr: FunctionCall, batch: VectorBatch) -> DataVector:
        name = expr.name.upper()
        args = [self.evaluate(a, batch) for a in expr.args]
        n = batch.row_count

        # String functions
        if name == 'UPPER':
            return self._apply_str1(args[0], n, lambda s: s.upper())
        if name == 'LOWER':
            return self._apply_str1(args[0], n, lambda s: s.lower())
        if name == 'LENGTH':
            return self._apply_num1(args[0], n, lambda s: len(str(s)), DataType.INT)
        if name == 'TRIM':
            return self._apply_str1(args[0], n, lambda s: s.strip())
        if name == 'LTRIM':
            return self._apply_str1(args[0], n, lambda s: s.lstrip())
        if name == 'RTRIM':
            return self._apply_str1(args[0], n, lambda s: s.rstrip())
        if name == 'REVERSE':
            return self._apply_str1(args[0], n, lambda s: s[::-1])
        if name == 'REPLACE' and len(args) >= 3:
            return self._apply_str3(args[0], args[1], args[2], n,
                                    lambda s, a, b: s.replace(a, b))
        if name in ('SUBSTR', 'SUBSTRING'):
            return self._eval_substr(args, n)
        if name in ('CONCAT', 'CONCAT_WS'):
            return self._eval_concat_func(name, args, n)
        if name == 'POSITION' and len(args) >= 2:
            return self._apply_num2(args[0], args[1], n,
                                    lambda sub, s: str(s).find(str(sub)) + 1, DataType.INT)
        if name == 'LEFT' and len(args) >= 2:
            return self._apply_str_int(args[0], args[1], n, lambda s, k: s[:k])
        if name == 'RIGHT' and len(args) >= 2:
            return self._apply_str_int(args[0], args[1], n,
                                       lambda s, k: s[-k:] if k > 0 else '')
        if name == 'REPEAT' and len(args) >= 2:
            return self._apply_str_int(args[0], args[1], n, lambda s, k: s * k)
        if name == 'STARTS_WITH' and len(args) >= 2:
            return self._apply_bool2(args[0], args[1], n,
                                     lambda s, p: str(s).startswith(str(p)))
        if name == 'ENDS_WITH' and len(args) >= 2:
            return self._apply_bool2(args[0], args[1], n,
                                     lambda s, p: str(s).endswith(str(p)))
        if name == 'CONTAINS' and len(args) >= 2:
            return self._apply_bool2(args[0], args[1], n,
                                     lambda s, p: str(p) in str(s))

        # Math functions
        import math
        if name == 'ABS':
            return self._apply_num1(args[0], n, lambda x: abs(x), args[0].dtype)
        if name in ('CEIL', 'CEILING'):
            return self._apply_num1(args[0], n, lambda x: math.ceil(x), DataType.BIGINT)
        if name == 'FLOOR':
            return self._apply_num1(args[0], n, lambda x: math.floor(x), DataType.BIGINT)
        if name == 'ROUND':
            if len(args) >= 2:
                return self._apply_num2(args[0], args[1], n,
                                        lambda x, d: round(x, int(d)), DataType.DOUBLE)
            return self._apply_num1(args[0], n, lambda x: round(x), DataType.BIGINT)
        if name == 'POWER' and len(args) >= 2:
            return self._apply_num2(args[0], args[1], n,
                                    lambda x, y: math.pow(x, y), DataType.DOUBLE)
        if name == 'SQRT':
            return self._apply_num1(args[0], n, lambda x: math.sqrt(x), DataType.DOUBLE)
        if name == 'MOD' and len(args) >= 2:
            return self._apply_num2(args[0], args[1], n,
                                    lambda x, y: x % y if y != 0 else None, args[0].dtype)
        if name == 'SIGN':
            return self._apply_num1(args[0], n,
                                    lambda x: (1 if x > 0 else -1 if x < 0 else 0), DataType.INT)
        if name == 'LN':
            return self._apply_num1(args[0], n, lambda x: math.log(x), DataType.DOUBLE)
        if name == 'LOG2':
            return self._apply_num1(args[0], n, lambda x: math.log2(x), DataType.DOUBLE)
        if name == 'LOG10':
            return self._apply_num1(args[0], n, lambda x: math.log10(x), DataType.DOUBLE)
        if name == 'EXP':
            return self._apply_num1(args[0], n, lambda x: math.exp(x), DataType.DOUBLE)
        if name == 'GREATEST' and len(args) >= 2:
            return self._apply_variadic(args, n, lambda vs: max(
                (v for v in vs if v is not None), default=None), args[0].dtype)
        if name == 'LEAST' and len(args) >= 2:
            return self._apply_variadic(args, n, lambda vs: min(
                (v for v in vs if v is not None), default=None), args[0].dtype)

        # Conditional
        if name == 'COALESCE':
            return self._eval_coalesce(args, n)
        if name == 'NULLIF' and len(args) >= 2:
            return self._eval_nullif(args[0], args[1], n)
        if name == 'IF' and len(args) >= 3:
            return self._eval_if_func(args[0], args[1], args[2], n)

        raise ExecutionError(f"unknown function: {expr.name}")

    # ==================================================================
    # Function helpers
    # ==================================================================
    def _apply_str1(self, v: DataVector, n: int, fn: Any) -> DataVector:
        rd: list = []
        rn = Bitmap(n)
        for i in range(n):
            if v.is_null(i):
                rn.set_bit(i)
                rd.append('')
            else:
                rd.append(fn(str(v.get(i))))
        return DataVector(dtype=DataType.VARCHAR, data=rd, nulls=rn, _length=n)

    def _apply_str3(self, a: DataVector, b: DataVector, c: DataVector,
                    n: int, fn: Any) -> DataVector:
        rd: list = []
        rn = Bitmap(n)
        for i in range(n):
            if a.is_null(i) or b.is_null(i) or c.is_null(i):
                rn.set_bit(i)
                rd.append('')
            else:
                rd.append(fn(str(a.get(i)), str(b.get(i)), str(c.get(i))))
        return DataVector(dtype=DataType.VARCHAR, data=rd, nulls=rn, _length=n)

    def _apply_str_int(self, sv: DataVector, iv: DataVector,
                       n: int, fn: Any) -> DataVector:
        rd: list = []
        rn = Bitmap(n)
        for i in range(n):
            if sv.is_null(i) or iv.is_null(i):
                rn.set_bit(i)
                rd.append('')
            else:
                rd.append(fn(str(sv.get(i)), int(iv.get(i))))
        return DataVector(dtype=DataType.VARCHAR, data=rd, nulls=rn, _length=n)

    def _apply_num1(self, v: DataVector, n: int, fn: Any,
                    dt: DataType) -> DataVector:
        code = DTYPE_TO_ARRAY_CODE.get(dt)
        rd: Any = TypedVector(code) if code else []
        rn = Bitmap(n)
        for i in range(n):
            if v.is_null(i):
                rn.set_bit(i)
                if isinstance(rd, TypedVector):
                    rd.append(0)
                else:
                    rd.append(None)
            else:
                val = fn(v.get(i))
                if val is None:
                    rn.set_bit(i)
                    val = 0 if isinstance(rd, TypedVector) else None
                if isinstance(rd, TypedVector):
                    rd.append(val)
                else:
                    rd.append(val)
        return DataVector(dtype=dt, data=rd, nulls=rn, _length=n)

    def _apply_num2(self, a: DataVector, b: DataVector, n: int,
                    fn: Any, dt: DataType) -> DataVector:
        code = DTYPE_TO_ARRAY_CODE.get(dt)
        rd: Any = TypedVector(code) if code else []
        rn = Bitmap(n)
        for i in range(n):
            if a.is_null(i) or b.is_null(i):
                rn.set_bit(i)
                if isinstance(rd, TypedVector):
                    rd.append(0)
                else:
                    rd.append(None)
            else:
                val = fn(a.get(i), b.get(i))
                if val is None:
                    rn.set_bit(i)
                    val = 0 if isinstance(rd, TypedVector) else None
                if isinstance(rd, TypedVector):
                    rd.append(val)
                else:
                    rd.append(val)
        return DataVector(dtype=dt, data=rd, nulls=rn, _length=n)

    def _apply_bool2(self, a: DataVector, b: DataVector, n: int,
                     fn: Any) -> DataVector:
        rd = Bitmap(n)
        rn = Bitmap(n)
        for i in range(n):
            if a.is_null(i) or b.is_null(i):
                rn.set_bit(i)
            elif fn(a.get(i), b.get(i)):
                rd.set_bit(i)
        return DataVector(dtype=DataType.BOOLEAN, data=rd, nulls=rn, _length=n)

    def _apply_variadic(self, args: list, n: int, fn: Any,
                        dt: DataType) -> DataVector:
        results: list = [None] * n
        for i in range(n):
            vals = [a.get(i) for a in args]
            results[i] = fn(vals)
        return self._list_to_vector(results, dt, n)

    def _eval_substr(self, args: list, n: int) -> DataVector:
        sv = args[0]
        rd: list = []
        rn = Bitmap(n)
        for i in range(n):
            if sv.is_null(i):
                rn.set_bit(i)
                rd.append('')
                continue
            s = str(sv.get(i))
            start = int(args[1].get(i)) - 1 if len(args) > 1 and not args[1].is_null(i) else 0
            if start < 0:
                start = 0
            if len(args) > 2 and not args[2].is_null(i):
                length = int(args[2].get(i))
                rd.append(s[start:start + length])
            else:
                rd.append(s[start:])
        return DataVector(dtype=DataType.VARCHAR, data=rd, nulls=rn, _length=n)

    def _eval_concat_func(self, name: str, args: list, n: int) -> DataVector:
        rd: list = []
        rn = Bitmap(n)
        for i in range(n):
            if name == 'CONCAT_WS':
                if args[0].is_null(i):
                    rn.set_bit(i)
                    rd.append('')
                    continue
                sep = str(args[0].get(i))
                parts = [str(a.get(i)) for a in args[1:] if not a.is_null(i)]
                rd.append(sep.join(parts))
            else:
                parts = []
                any_null = False
                for a in args:
                    if a.is_null(i):
                        any_null = True
                        break
                    parts.append(str(a.get(i)))
                if any_null:
                    rn.set_bit(i)
                    rd.append('')
                else:
                    rd.append(''.join(parts))
        return DataVector(dtype=DataType.VARCHAR, data=rd, nulls=rn, _length=n)

    def _eval_coalesce(self, args: list, n: int) -> DataVector:
        # Determine type from first arg that has actual non-null values
        dt = DataType.UNKNOWN
        for a in args:
            has_val = any(not a.is_null(i) for i in range(len(a))) if len(a) > 0 else False
            if has_val:
                dt = a.dtype
                break
        if dt == DataType.UNKNOWN:
            # Fallback: first non-UNKNOWN dtype
            for a in args:
                if a.dtype != DataType.UNKNOWN:
                    dt = a.dtype
                    break
        if dt == DataType.UNKNOWN:
            dt = DataType.INT
        results: list = [None] * n
        for i in range(n):
            for a in args:
                if not a.is_null(i):
                    results[i] = a.get(i)
                    break
        return self._list_to_vector(results, dt, n)

    def _eval_nullif(self, a: DataVector, b: DataVector, n: int) -> DataVector:
        results: list = [None] * n
        for i in range(n):
            if a.is_null(i):
                continue
            if b.is_null(i):
                results[i] = a.get(i)
                continue
            results[i] = None if a.get(i) == b.get(i) else a.get(i)
        return self._list_to_vector(results, a.dtype, n)

    def _eval_if_func(self, cond: DataVector, then: DataVector,
                      else_: DataVector, n: int) -> DataVector:
        results: list = [None] * n
        for i in range(n):
            if cond.is_null(i) or not cond.get(i):
                results[i] = else_.get(i)
            else:
                results[i] = then.get(i)
        dt = then.dtype if then.dtype != DataType.UNKNOWN else else_.dtype
        return self._list_to_vector(results, dt, n)

    # ==================================================================
    # Helpers
    # ==================================================================
    def _to_bool(self, vec: DataVector) -> DataVector:
        if vec.dtype == DataType.BOOLEAN:
            return vec
        if is_numeric(vec.dtype):
            n = len(vec)
            d = Bitmap(n)
            nl = Bitmap(n)
            for i in range(n):
                if vec.is_null(i):
                    nl.set_bit(i)
                elif vec.get(i) != 0:
                    d.set_bit(i)
            return DataVector(dtype=DataType.BOOLEAN, data=d, nulls=nl, _length=n)
        raise ExecutionError(f"cannot convert {vec.dtype.name} to BOOLEAN")

    def _bool_to_bitmap(self, vec: DataVector) -> Bitmap:
        n = len(vec)
        bm = Bitmap(n)
        assert isinstance(vec.data, Bitmap)
        for i in range(n):
            if not vec.is_null(i) and vec.data.get_bit(i):
                bm.set_bit(i)
        return bm

    def _list_to_vector(self, values: list, dtype: DataType, n: int) -> DataVector:
        """Convert Python list (with Nones) to DataVector."""
        if dtype == DataType.UNKNOWN:
            dtype = DataType.INT
        code = DTYPE_TO_ARRAY_CODE.get(dtype)
        nulls = Bitmap(n)
        if dtype == DataType.BOOLEAN:
            data: Any = Bitmap(n)
            for i in range(n):
                if values[i] is None:
                    nulls.set_bit(i)
                elif values[i]:
                    data.set_bit(i)
        elif dtype in (DataType.VARCHAR, DataType.TEXT):
            data = []
            for i in range(n):
                if values[i] is None:
                    nulls.set_bit(i)
                    data.append('')
                else:
                    data.append(str(values[i]))
        elif code is not None:
            data = TypedVector(code)
            for i in range(n):
                if values[i] is None:
                    nulls.set_bit(i)
                    data.append(0)
                else:
                    data.append(values[i])
        else:
            data = TypedVector('q')
            for i in range(n):
                if values[i] is None:
                    nulls.set_bit(i)
                    data.append(0)
                else:
                    data.append(values[i])
        return DataVector(dtype=dtype, data=data, nulls=nulls, _length=n)

    # ==================================================================
    # Static type inference
    # ==================================================================
    @staticmethod
    def infer_type(expr: Any, input_schema: Dict[str, DataType]) -> DataType:
        if isinstance(expr, Literal):
            return expr.inferred_type if expr.inferred_type != DataType.UNKNOWN else DataType.INT
        if isinstance(expr, ColumnRef):
            # Try qualified
            if expr.table:
                q = f"{expr.table}.{expr.column}"
                if q in input_schema:
                    return input_schema[q]
            return input_schema.get(expr.column, DataType.UNKNOWN)
        if isinstance(expr, AliasExpr):
            return ExpressionEvaluator.infer_type(expr.expr, input_schema)
        if isinstance(expr, BinaryExpr):
            if expr.op in ('+', '-', '*', '/', '%'):
                lt = ExpressionEvaluator.infer_type(expr.left, input_schema)
                rt = ExpressionEvaluator.infer_type(expr.right, input_schema)
                try:
                    return promote(lt, rt)
                except TypeMismatchError:
                    return DataType.UNKNOWN
            if expr.op == '||':
                return DataType.VARCHAR
            return DataType.BOOLEAN
        if isinstance(expr, UnaryExpr):
            if expr.op == 'NOT':
                return DataType.BOOLEAN
            return ExpressionEvaluator.infer_type(expr.operand, input_schema)
        if isinstance(expr, IsNullExpr):
            return DataType.BOOLEAN
        if isinstance(expr, (InExpr, BetweenExpr, LikeExpr)):
            return DataType.BOOLEAN
        if isinstance(expr, CaseExpr):
            if expr.when_clauses:
                return ExpressionEvaluator.infer_type(expr.when_clauses[0][1], input_schema)
            return DataType.UNKNOWN
        if isinstance(expr, CastExpr):
            if expr.type_name:
                dt, _ = resolve_type_name(expr.type_name.name, expr.type_name.params)
                return dt
            return DataType.UNKNOWN
        if isinstance(expr, FunctionCall):
            name = expr.name.upper()
            if name in ('UPPER', 'LOWER', 'TRIM', 'LTRIM', 'RTRIM', 'REVERSE', 'REPLACE',
                         'SUBSTR', 'SUBSTRING', 'CONCAT', 'CONCAT_WS', 'LEFT', 'RIGHT',
                         'REPEAT', 'COALESCE'):
                if name == 'COALESCE' and expr.args:
                    return ExpressionEvaluator.infer_type(expr.args[0], input_schema)
                return DataType.VARCHAR
            if name in ('LENGTH', 'SIGN', 'POSITION'):
                return DataType.INT
            if name == 'ABS' and expr.args:
                return ExpressionEvaluator.infer_type(expr.args[0], input_schema)
            if name in ('CEIL', 'CEILING', 'FLOOR') or (name == 'ROUND' and len(expr.args) < 2):
                return DataType.BIGINT
            if name in ('POWER', 'SQRT', 'LN', 'LOG2', 'LOG10', 'EXP', 'ROUND'):
                return DataType.DOUBLE
            if name in ('STARTS_WITH', 'ENDS_WITH', 'CONTAINS'):
                return DataType.BOOLEAN
            if name == 'NULLIF' and expr.args:
                return ExpressionEvaluator.infer_type(expr.args[0], input_schema)
            if name == 'IF' and len(expr.args) >= 2:
                return ExpressionEvaluator.infer_type(expr.args[1], input_schema)
            return DataType.UNKNOWN
        if isinstance(expr, AggregateCall):
            upper = expr.name.upper()
            if upper == 'COUNT':
                return DataType.BIGINT
            if upper == 'AVG':
                return DataType.DOUBLE
            if upper == 'SUM':
                if expr.args and not isinstance(expr.args[0], StarExpr):
                    at = ExpressionEvaluator.infer_type(expr.args[0], input_schema)
                    if at in (DataType.FLOAT, DataType.DOUBLE):
                        return DataType.DOUBLE
                return DataType.BIGINT
            if upper in ('MIN', 'MAX'):
                if expr.args and not isinstance(expr.args[0], StarExpr):
                    return ExpressionEvaluator.infer_type(expr.args[0], input_schema)
            return DataType.UNKNOWN
        return DataType.UNKNOWN

    # ==================================================================
    # Type casting
    # ==================================================================
    @staticmethod
    def cast_vector(vec: DataVector, target: DataType) -> DataVector:
        if vec.dtype == target:
            return vec
        n = len(vec)
        if vec.dtype == DataType.BOOLEAN and is_numeric(target):
            code = DTYPE_TO_ARRAY_CODE.get(target, 'q')
            assert code is not None
            data = TypedVector(code)
            nulls = Bitmap(n)
            for i in range(n):
                if vec.is_null(i):
                    nulls.set_bit(i)
                    data.append(0)
                else:
                    data.append(1 if vec.get(i) else 0)
            return DataVector(dtype=target, data=data, nulls=nulls, _length=n)
        src_rank = {DataType.INT: 0, DataType.BIGINT: 1, DataType.FLOAT: 2, DataType.DOUBLE: 3}
        if vec.dtype in src_rank and target in src_rank:
            if src_rank[target] >= src_rank[vec.dtype]:
                code = DTYPE_TO_ARRAY_CODE.get(target, 'q')
                assert code is not None
                data = TypedVector(code)
                nulls = Bitmap(n)
                for i in range(n):
                    if vec.is_null(i):
                        nulls.set_bit(i)
                        data.append(0)
                    else:
                        v = vec.get(i)
                        data.append(float(v) if target in (DataType.FLOAT, DataType.DOUBLE) else int(v))
                return DataVector(dtype=target, data=data, nulls=nulls, _length=n)
        raise TypeMismatchError(f"cannot cast {vec.dtype.name} to {target.name}",
                                expected=target.name, actual=vec.dtype.name)


def _like_match(text: str, pattern: str) -> bool:
    """SQL LIKE: % = any string, _ = any char."""
    regex = '^'
    for ch in pattern:
        if ch == '%':
            regex += '.*'
        elif ch == '_':
            regex += '.'
        elif ch in r'\.^$+?{}[]|()':
            regex += '\\' + ch
        else:
            regex += ch
    regex += '$'
    return bool(_re.match(regex, text, _re.DOTALL))
