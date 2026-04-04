from __future__ import annotations
"""Expression evaluator — vectorised evaluation over DataVectors."""

import operator as _op
from typing import Any, Dict, Optional

from executor.core.vector import DataVector
from executor.core.batch import VectorBatch
from metal.bitmap import Bitmap
from metal.typed_vector import TypedVector
from parser.ast import (
    AggregateCall, AliasExpr, BinaryExpr, ColumnRef, FunctionCall,
    IsNullExpr, Literal, StarExpr, UnaryExpr,
)
from storage.types import (
    DTYPE_TO_ARRAY_CODE, DataType, is_numeric, is_string, promote,
)
from utils.errors import (
    DivisionByZeroError, ExecutionError, TypeMismatchError,
)


class ExpressionEvaluator:
    """Evaluates AST expression nodes over VectorBatch data."""

    def __init__(self, registry: Optional[Any] = None) -> None:
        self._registry = registry

    # ==================================================================
    # Public API
    # ==================================================================
    def evaluate(self, expr: Any, batch: VectorBatch) -> DataVector:
        if isinstance(expr, Literal):
            return self._eval_literal(expr, batch.row_count)
        if isinstance(expr, ColumnRef):
            return batch.get_column(expr.column)
        if isinstance(expr, AliasExpr):
            return self.evaluate(expr.expr, batch)
        if isinstance(expr, BinaryExpr):
            left = self.evaluate(expr.left, batch)
            right = self.evaluate(expr.right, batch)
            return self._eval_binary(expr.op, left, right)
        if isinstance(expr, UnaryExpr):
            return self._eval_unary(expr, batch)
        if isinstance(expr, IsNullExpr):
            return self._eval_is_null(expr, batch)
        if isinstance(expr, AggregateCall):
            raise ExecutionError("aggregate in non-aggregate context")
        if isinstance(expr, FunctionCall):
            raise ExecutionError(f"scalar function '{expr.name}' is not supported in Phase 1")
        if isinstance(expr, StarExpr):
            raise ExecutionError("internal: StarExpr in eval")
        raise ExecutionError(f"unknown expression: {type(expr).__name__}")

    def evaluate_predicate(self, expr: Any, batch: VectorBatch) -> Bitmap:
        vec = self.evaluate(expr, batch)
        if vec.dtype == DataType.BOOLEAN:
            return self._boolean_vec_to_bitmap(vec)
        if is_numeric(vec.dtype):
            bm = Bitmap(len(vec))
            for i in range(len(vec)):
                if not vec.is_null(i) and vec.get(i) != 0:
                    bm.set_bit(i)
            return bm
        raise ExecutionError(f"WHERE clause must be boolean, got {vec.dtype.name}")

    # ==================================================================
    # Literal
    # ==================================================================
    def _eval_literal(self, lit: Literal, row_count: int) -> DataVector:
        if lit.value is None:
            dt = lit.inferred_type if lit.inferred_type != DataType.UNKNOWN else DataType.INT
            nulls = Bitmap(row_count)
            for i in range(row_count):
                nulls.set_bit(i)
            return DataVector.from_nulls(dt, row_count, nulls)

        dtype = lit.inferred_type
        if dtype == DataType.UNKNOWN:
            dtype = DataType.INT

        code = DTYPE_TO_ARRAY_CODE.get(dtype)
        if code is not None:
            data: Any = TypedVector(code, [lit.value] * row_count)
        elif dtype in (DataType.VARCHAR, DataType.TEXT):
            data = [lit.value] * row_count
        elif dtype == DataType.BOOLEAN:
            data = Bitmap(row_count)
            if lit.value:
                for i in range(row_count):
                    data.set_bit(i)
        else:
            data = TypedVector('q', [lit.value] * row_count)

        return DataVector(dtype=dtype, data=data, nulls=Bitmap(row_count), _length=row_count)

    # ==================================================================
    # Binary
    # ==================================================================
    def _eval_binary(self, op: str, left: DataVector, right: DataVector) -> DataVector:
        if op in ('+', '-', '*', '/', '%'):
            return self._eval_arithmetic(op, left, right)
        if op in ('=', '!=', '<', '>', '<=', '>='):
            return self._eval_comparison(op, left, right)
        if op == '||':
            return self._eval_concat(left, right)
        if op == 'AND':
            return self._eval_and(self._to_boolean(left), self._to_boolean(right))
        if op == 'OR':
            return self._eval_or(self._to_boolean(left), self._to_boolean(right))
        raise ExecutionError(f"unsupported binary operator: {op}")

    # -- arithmetic ----------------------------------------------------
    def _eval_arithmetic(self, op: str, left: DataVector, right: DataVector) -> DataVector:
        target = promote(left.dtype, right.dtype)
        lv = self.cast_vector(left, target)
        rv = self.cast_vector(right, target)
        n = len(lv)

        ops = {'+': _op.add, '-': _op.sub, '*': _op.mul}
        is_int = target in (DataType.INT, DataType.BIGINT, DataType.BOOLEAN)

        code = DTYPE_TO_ARRAY_CODE.get(target)
        if code is not None:
            result_data: Any = TypedVector(code)
        else:
            result_data = []
        result_nulls = Bitmap(n)

        for i in range(n):
            if lv.is_null(i) or rv.is_null(i):
                result_nulls.set_bit(i)
                if isinstance(result_data, TypedVector):
                    result_data.append(0)
                else:
                    result_data.append(None)
                continue
            a = lv.get(i)
            b = rv.get(i)
            if op in ops:
                val = ops[op](a, b)
            elif op == '/':
                if b == 0:
                    if is_int:
                        raise DivisionByZeroError()
                    else:
                        val = float('inf') if a >= 0 else float('-inf')
                else:
                    if is_int:
                        val = int(a / b)  # truncate towards zero
                    else:
                        val = a / b
            elif op == '%':
                if b == 0:
                    raise DivisionByZeroError()
                val = a % b
            else:
                raise ExecutionError(f"unknown arithmetic op: {op}")

            # Range check for INT
            if target == DataType.INT:
                if not (-2_147_483_648 <= val <= 2_147_483_647):
                    from utils.errors import NumericOverflowError
                    raise NumericOverflowError(f"integer overflow: {val}")

            if isinstance(result_data, TypedVector):
                result_data.append(val)
            else:
                result_data.append(val)

        return DataVector(dtype=target, data=result_data, nulls=result_nulls, _length=n)

    # -- comparison ----------------------------------------------------
    def _eval_comparison(self, op: str, left: DataVector, right: DataVector) -> DataVector:
        # Promote for consistent comparison
        if is_numeric(left.dtype) and is_numeric(right.dtype):
            target = promote(left.dtype, right.dtype)
            lv = self.cast_vector(left, target)
            rv = self.cast_vector(right, target)
        elif is_string(left.dtype) and is_string(right.dtype):
            lv, rv = left, right
        elif left.dtype == DataType.BOOLEAN and right.dtype == DataType.BOOLEAN:
            lv, rv = left, right
        else:
            # Try promotion anyway
            try:
                target = promote(left.dtype, right.dtype)
                lv = self.cast_vector(left, target)
                rv = self.cast_vector(right, target)
            except TypeMismatchError:
                lv, rv = left, right

        n = len(lv)
        cmp_ops = {'=': _op.eq, '!=': _op.ne, '<': _op.lt,
                   '>': _op.gt, '<=': _op.le, '>=': _op.ge}
        fn = cmp_ops[op]

        result_data = Bitmap(n)
        result_nulls = Bitmap(n)

        for i in range(n):
            if lv.is_null(i) or rv.is_null(i):
                result_nulls.set_bit(i)
                continue
            if fn(lv.get(i), rv.get(i)):
                result_data.set_bit(i)

        return DataVector(dtype=DataType.BOOLEAN, data=result_data,
                          nulls=result_nulls, _length=n)

    # -- concat --------------------------------------------------------
    def _eval_concat(self, left: DataVector, right: DataVector) -> DataVector:
        n = len(left)
        result_data: list[str] = []
        result_nulls = Bitmap(n)
        for i in range(n):
            if left.is_null(i) or right.is_null(i):
                result_nulls.set_bit(i)
                result_data.append('')
            else:
                result_data.append(str(left.get(i)) + str(right.get(i)))
        return DataVector(dtype=DataType.VARCHAR, data=result_data,
                          nulls=result_nulls, _length=n)

    # -- logical -------------------------------------------------------
    def _eval_and(self, left: DataVector, right: DataVector) -> DataVector:
        n = len(left)
        result_data = Bitmap(n)
        result_nulls = Bitmap(n)
        for i in range(n):
            l_null = left.is_null(i)
            r_null = right.is_null(i)
            l_val = left.get(i) if not l_null else None
            r_val = right.get(i) if not r_null else None

            if not l_null and not l_val:
                # FALSE AND anything → FALSE
                pass
            elif not r_null and not r_val:
                # anything AND FALSE → FALSE
                pass
            elif l_null or r_null:
                result_nulls.set_bit(i)
            else:
                if l_val and r_val:
                    result_data.set_bit(i)
        return DataVector(dtype=DataType.BOOLEAN, data=result_data,
                          nulls=result_nulls, _length=n)

    def _eval_or(self, left: DataVector, right: DataVector) -> DataVector:
        n = len(left)
        result_data = Bitmap(n)
        result_nulls = Bitmap(n)
        for i in range(n):
            l_null = left.is_null(i)
            r_null = right.is_null(i)
            l_val = left.get(i) if not l_null else None
            r_val = right.get(i) if not r_null else None

            if not l_null and l_val:
                # TRUE OR anything → TRUE
                result_data.set_bit(i)
            elif not r_null and r_val:
                # anything OR TRUE → TRUE
                result_data.set_bit(i)
            elif l_null or r_null:
                result_nulls.set_bit(i)
            # else both FALSE → FALSE (no bit set)
        return DataVector(dtype=DataType.BOOLEAN, data=result_data,
                          nulls=result_nulls, _length=n)

    # ==================================================================
    # Unary
    # ==================================================================
    def _eval_unary(self, expr: Any, batch: VectorBatch) -> DataVector:
        operand = self.evaluate(expr.operand, batch)

        if expr.op == '+':
            return operand

        if expr.op == '-':
            n = len(operand)
            if not is_numeric(operand.dtype):
                raise ExecutionError(f"cannot negate {operand.dtype.name}")
            code = DTYPE_TO_ARRAY_CODE.get(operand.dtype)
            if code is None:
                raise ExecutionError(f"cannot negate {operand.dtype.name}")
            result_data = TypedVector(code)
            result_nulls = Bitmap(n)
            for i in range(n):
                if operand.is_null(i):
                    result_nulls.set_bit(i)
                    result_data.append(0)
                else:
                    result_data.append(-operand.get(i))
            return DataVector(dtype=operand.dtype, data=result_data,
                              nulls=result_nulls, _length=n)

        if expr.op == 'NOT':
            bvec = self._to_boolean(operand)
            n = len(bvec)
            result_data = Bitmap(n)
            result_nulls = Bitmap(n)
            for i in range(n):
                if bvec.is_null(i):
                    result_nulls.set_bit(i)
                elif not bvec.data.get_bit(i):
                    result_data.set_bit(i)
            return DataVector(dtype=DataType.BOOLEAN, data=result_data,
                              nulls=result_nulls, _length=n)

        raise ExecutionError(f"unsupported unary operator: {expr.op}")

    # ==================================================================
    # IS NULL
    # ==================================================================
    def _eval_is_null(self, expr: Any, batch: VectorBatch) -> DataVector:
        operand = self.evaluate(expr.expr, batch)
        n = len(operand)
        result_data = Bitmap(n)
        for i in range(n):
            is_n = operand.is_null(i)
            if (is_n and not expr.negated) or (not is_n and expr.negated):
                result_data.set_bit(i)
        return DataVector(dtype=DataType.BOOLEAN, data=result_data,
                          nulls=Bitmap(n), _length=n)

    # ==================================================================
    # Helpers
    # ==================================================================
    def _to_boolean(self, vec: DataVector) -> DataVector:
        if vec.dtype == DataType.BOOLEAN:
            return vec
        if is_numeric(vec.dtype):
            n = len(vec)
            data = Bitmap(n)
            nulls = Bitmap(n)
            for i in range(n):
                if vec.is_null(i):
                    nulls.set_bit(i)
                elif vec.get(i) != 0:
                    data.set_bit(i)
            return DataVector(dtype=DataType.BOOLEAN, data=data, nulls=nulls, _length=n)
        raise ExecutionError(f"cannot convert {vec.dtype.name} to BOOLEAN")

    def _boolean_vec_to_bitmap(self, vec: DataVector) -> Bitmap:
        """Convert a BOOLEAN DataVector to a Bitmap.  NULL → FALSE."""
        n = len(vec)
        bm = Bitmap(n)
        assert isinstance(vec.data, Bitmap)
        for i in range(n):
            if not vec.is_null(i) and vec.data.get_bit(i):
                bm.set_bit(i)
        return bm

    # ==================================================================
    # Static type inference
    # ==================================================================
    @staticmethod
    def infer_type(expr: Any, input_schema: Dict[str, DataType]) -> DataType:
        if isinstance(expr, Literal):
            return expr.inferred_type if expr.inferred_type != DataType.UNKNOWN else DataType.INT
        if isinstance(expr, ColumnRef):
            if expr.column in input_schema:
                return input_schema[expr.column]
            return DataType.UNKNOWN
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
            return DataType.BOOLEAN  # comparisons / AND / OR
        if isinstance(expr, UnaryExpr):
            if expr.op == 'NOT':
                return DataType.BOOLEAN
            return ExpressionEvaluator.infer_type(expr.operand, input_schema)
        if isinstance(expr, IsNullExpr):
            return DataType.BOOLEAN
        if isinstance(expr, AggregateCall):
            # Best-effort type inference for aggregates
            upper = expr.name.upper()
            if upper == 'COUNT':
                return DataType.BIGINT
            if upper == 'AVG':
                return DataType.DOUBLE
            if upper in ('SUM',):
                if expr.args and not isinstance(expr.args[0], StarExpr):
                    at = ExpressionEvaluator.infer_type(expr.args[0], input_schema)
                    if at in (DataType.FLOAT, DataType.DOUBLE):
                        return DataType.DOUBLE
                return DataType.BIGINT
            if upper in ('MIN', 'MAX'):
                if expr.args and not isinstance(expr.args[0], StarExpr):
                    return ExpressionEvaluator.infer_type(expr.args[0], input_schema)
            return DataType.UNKNOWN
        if isinstance(expr, StarExpr):
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

        # BOOLEAN → numeric
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

        # Numeric widening
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
                        val = vec.get(i)
                        if target in (DataType.FLOAT, DataType.DOUBLE):
                            val = float(val)
                        else:
                            val = int(val)
                        data.append(val)
                return DataVector(dtype=target, data=data, nulls=nulls, _length=n)

        raise TypeMismatchError(
            f"cannot cast {vec.dtype.name} to {target.name}",
            expected=target.name, actual=vec.dtype.name,
        )
