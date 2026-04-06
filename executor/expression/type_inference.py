from __future__ import annotations
"""类型推断 + 向量类型转换。"""
from typing import Any, Dict
from metal.bitmap import Bitmap
from metal.typed_vector import TypedVector
from storage.types import (
    DTYPE_TO_ARRAY_CODE, DataType, is_numeric,
    promote, resolve_type_name, _NUMERIC_RANK)
from utils.errors import TypeMismatchError
from parser.ast import (
    AggregateCall, AliasExpr, BetweenExpr, BinaryExpr,
    CaseExpr, CastExpr, ColumnRef, FunctionCall, InExpr,
    IsNullExpr, LikeExpr, Literal, StarExpr, UnaryExpr)

# 函数返回类型分组（避免每次调用创建临时集合）
INFER_STR = frozenset({
    'UPPER', 'LOWER', 'TRIM', 'LTRIM', 'RTRIM', 'REVERSE',
    'REPLACE', 'SUBSTR', 'SUBSTRING', 'CONCAT', 'CONCAT_WS',
    'LEFT', 'RIGHT', 'REPEAT', 'LPAD', 'RPAD', 'INITCAP',
    'CHR', 'SPLIT_PART', 'REGEXP_REPLACE', 'REGEXP_EXTRACT',
    'SPLIT', 'ENCODE', 'DECODE', 'DATE_FORMAT',
    'ARRAY', 'ARRAY_CREATE', 'ARRAY_SORT', 'ARRAY_REVERSE',
    'ARRAY_DISTINCT', 'ARRAY_FLATTEN', 'ARRAY_INTERSECT',
    'ARRAY_UNION', 'ARRAY_EXCEPT', 'ARRAY_APPEND',
    'ARRAY_PREPEND', 'ARRAY_REMOVE', 'ARRAY_CONCAT',
    'ARRAY_SLICE', 'ARRAY_JOIN', 'GENERATE_SERIES',
    'EXPLODE', 'UNNEST', 'TRY_CAST'})

INFER_INT = frozenset({
    'LENGTH', 'SIGN', 'POSITION', 'ASCII', 'YEAR', 'MONTH',
    'DAY', 'HOUR', 'MINUTE', 'SECOND', 'DAY_OF_WEEK',
    'DAY_OF_YEAR', 'WEEK_OF_YEAR', 'QUARTER', 'DATE_DIFF',
    'DATE_ADD', 'DATE_SUB', 'TRUNC', 'TRUNCATE',
    'WIDTH_BUCKET', 'BIT_COUNT', 'ARRAY_LENGTH',
    'ARRAY_POSITION'})

INFER_BIGINT = frozenset({'CEIL', 'CEILING', 'FLOOR'})

INFER_DOUBLE = frozenset({
    'POWER', 'SQRT', 'CBRT', 'LN', 'LOG', 'LOG2', 'LOG10',
    'EXP', 'RANDOM', 'JACCARD_SIMILARITY',
    'COSINE_SIMILARITY'})

INFER_BOOL = frozenset({
    'STARTS_WITH', 'ENDS_WITH', 'CONTAINS',
    'REGEXP_MATCH', 'ARRAY_CONTAINS'})


def infer_type(expr: Any,
               schema: Dict[str, DataType]) -> DataType:
    """静态类型推断。"""
    if isinstance(expr, Literal):
        return (expr.inferred_type
                if expr.inferred_type != DataType.UNKNOWN
                else DataType.INT)
    if isinstance(expr, ColumnRef):
        if expr.table:
            q = f"{expr.table}.{expr.column}"
            if q in schema:
                return schema[q]
        return schema.get(expr.column, DataType.UNKNOWN)
    if isinstance(expr, AliasExpr):
        return infer_type(expr.expr, schema)
    if isinstance(expr, BinaryExpr):
        if expr.op in ('+', '-', '*', '/', '%'):
            lt = infer_type(expr.left, schema)
            rt = infer_type(expr.right, schema)
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
        return infer_type(expr.operand, schema)
    if isinstance(expr, IsNullExpr):
        return DataType.BOOLEAN
    if isinstance(expr, (InExpr, BetweenExpr, LikeExpr)):
        return DataType.BOOLEAN
    if isinstance(expr, CaseExpr):
        if expr.when_clauses:
            return infer_type(
                expr.when_clauses[0][1], schema)
        return DataType.UNKNOWN
    if isinstance(expr, CastExpr):
        if expr.type_name:
            dt, _ = resolve_type_name(
                expr.type_name.name, expr.type_name.params)
            return dt
        return DataType.UNKNOWN
    if isinstance(expr, FunctionCall):
        nm = expr.name.upper()
        if nm in INFER_STR: return DataType.VARCHAR
        if nm in INFER_INT: return DataType.INT
        if nm in INFER_BIGINT: return DataType.BIGINT
        if nm in INFER_DOUBLE: return DataType.DOUBLE
        if nm in INFER_BOOL: return DataType.BOOLEAN
        if nm == 'ROUND':
            return (DataType.DOUBLE if len(expr.args) >= 2
                    else DataType.BIGINT)
        if nm in ('NOW', 'CURRENT_TIMESTAMP'):
            return DataType.TIMESTAMP
        if nm == 'CURRENT_DATE':
            return DataType.DATE
        if nm in ('DATE_TRUNC', 'TO_DATE'):
            return DataType.DATE
        if nm == 'TO_TIMESTAMP':
            return DataType.TIMESTAMP
        if nm in ('EPOCH', 'HASH', 'MURMUR_HASH'):
            return DataType.BIGINT
        if nm == 'TYPEOF':
            return DataType.VARCHAR
        for passthrough in ('ABS', 'COALESCE', 'NULLIF',
                            'GREATEST', 'LEAST', 'MOD'):
            if nm == passthrough and expr.args:
                return infer_type(expr.args[0], schema)
        if nm == 'IF' and len(expr.args) >= 2:
            return infer_type(expr.args[1], schema)
        return DataType.UNKNOWN
    if isinstance(expr, AggregateCall):
        u = expr.name.upper()
        if u == 'COUNT':
            return DataType.BIGINT
        if u in ('AVG', 'AVG_DISTINCT', 'STDDEV',
                 'STDDEV_POP', 'VARIANCE', 'VAR_POP',
                 'APPROX_PERCENTILE'):
            return DataType.DOUBLE
        if u in ('SUM', 'SUM_DISTINCT'):
            if (expr.args
                    and not isinstance(expr.args[0], StarExpr)):
                at = infer_type(expr.args[0], schema)
                if at in (DataType.FLOAT, DataType.DOUBLE):
                    return DataType.DOUBLE
            return DataType.BIGINT
        if u in ('MIN', 'MAX', 'PERCENTILE_DISC', 'MODE'):
            if (expr.args
                    and not isinstance(expr.args[0], StarExpr)):
                return infer_type(expr.args[0], schema)
        if u in ('APPROX_COUNT_DISTINCT', 'COUNT_DISTINCT'):
            return DataType.BIGINT
        if u in ('MEDIAN', 'PERCENTILE_CONT'):
            return DataType.DOUBLE
        if u in ('STRING_AGG', 'ARRAY_AGG',
                 'APPROX_TOP_K', 'GROUPING'):
            return DataType.VARCHAR
        return DataType.UNKNOWN
    return DataType.UNKNOWN


def cast_vector(vec, target: DataType):
    """向量类型转换。"""
    from executor.core.vector import DataVector
    if vec.dtype == target:
        return vec
    n = len(vec)
    # BOOLEAN → 数值
    if vec.dtype == DataType.BOOLEAN and is_numeric(target):
        code = DTYPE_TO_ARRAY_CODE.get(target, 'q')
        assert code
        data = TypedVector(code)
        nulls = Bitmap(n)
        for i in range(n):
            if vec.is_null(i):
                nulls.set_bit(i)
                data.append(0)
            else:
                data.append(1 if vec.get(i) else 0)
        return DataVector(
            dtype=target, data=data,
            nulls=nulls, _length=n)
    # 数值提升
    vr = _NUMERIC_RANK.get(vec.dtype)
    tr = _NUMERIC_RANK.get(target)
    if vr is not None and tr is not None and tr >= vr:
        code = DTYPE_TO_ARRAY_CODE.get(target, 'q')
        assert code
        data = TypedVector(code)
        nulls = Bitmap(n)
        is_float = target in (DataType.FLOAT, DataType.DOUBLE)
        for i in range(n):
            if vec.is_null(i):
                nulls.set_bit(i)
                data.append(0)
            else:
                data.append(
                    float(vec.get(i)) if is_float
                    else int(vec.get(i)))
        return DataVector(
            dtype=target, data=data,
            nulls=nulls, _length=n)
    raise TypeMismatchError(
        f"无法转换 {vec.dtype.name} 为 {target.name}",
        expected=target.name, actual=vec.dtype.name)
