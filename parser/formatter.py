from __future__ import annotations
"""AST → SQL 文本格式化。用于 EXPLAIN 输出和调试。"""
from parser.ast import *
from parser.precedence import Precedence

# 运算符优先级（格式化时决定是否加括号）
_P = {
    'OR': 1, 'AND': 2, '=': 4, '!=': 4,
    '<': 4, '>': 4, '<=': 4, '>=': 4,
    '||': 5, '+': 6, '-': 6, '*': 7, '/': 7, '%': 7,
}


class Formatter:
    """AST 表达式 → SQL 字符串。"""

    @staticmethod
    def expr_to_sql(e: object) -> str:
        return Formatter._f(e, 0)

    @staticmethod
    def _f(n: object, pp: int) -> str:
        if isinstance(n, ColumnRef):
            return (f"{n.table}.{n.column}"
                    if n.table else n.column)
        if isinstance(n, Literal):
            if n.value is None:
                return 'NULL'
            if isinstance(n.value, bool):
                return 'TRUE' if n.value else 'FALSE'
            if isinstance(n.value, str):
                return f"'{n.value}'"
            return str(n.value)
        if isinstance(n, BinaryExpr):
            p = _P.get(n.op, 0)
            s = (f"{Formatter._f(n.left, p)} {n.op} "
                 f"{Formatter._f(n.right, p)}")
            return f"({s})" if p < pp else s
        if isinstance(n, UnaryExpr):
            o = Formatter._f(n.operand, Precedence.UNARY)
            return f"NOT {o}" if n.op == 'NOT' else f"{n.op}{o}"
        if isinstance(n, AggregateCall):
            if n.args and isinstance(n.args[0], StarExpr):
                return f"{n.name}(*)"
            a = ', '.join(
                Formatter._f(x, 0) for x in n.args)
            d = 'DISTINCT ' if n.distinct else ''
            return f"{n.name}({d}{a})"
        if isinstance(n, FunctionCall):
            a = ', '.join(
                Formatter._f(x, 0) for x in n.args)
            return f"{n.name}({a})"
        if isinstance(n, WindowCall):
            fn = Formatter._f(n.func, 0)
            parts = []
            if n.partition_by:
                parts.append('PARTITION BY ' + ', '.join(
                    Formatter._f(x, 0) for x in n.partition_by))
            if n.order_by:
                parts.append('ORDER BY ' + ', '.join(
                    Formatter._f(sk.expr, 0)
                    for sk in n.order_by))
            return f"{fn} OVER ({' '.join(parts)})"
        if isinstance(n, IsNullExpr):
            neg = 'NOT ' if n.negated else ''
            return f"{Formatter._f(n.expr, 0)} IS {neg}NULL"
        if isinstance(n, AliasExpr):
            return Formatter._f(n.expr, pp)
        if isinstance(n, StarExpr):
            return f"{n.table}.*" if n.table else '*'
        if isinstance(n, CaseExpr):
            return 'CASE ... END'
        if isinstance(n, CastExpr):
            tn = n.type_name.name if n.type_name else '?'
            return f"CAST({Formatter._f(n.expr, 0)} AS {tn})"
        if isinstance(n, InExpr):
            neg = ' NOT' if n.negated else ''
            return f"{Formatter._f(n.expr, 0)}{neg} IN (...)"
        if isinstance(n, BetweenExpr):
            neg = ' NOT' if n.negated else ''
            return (f"{Formatter._f(n.expr, 0)}"
                    f"{neg} BETWEEN ...")
        if isinstance(n, LikeExpr):
            neg = ' NOT' if n.negated else ''
            return (f"{Formatter._f(n.expr, 0)}"
                    f"{neg} LIKE ...")
        if isinstance(n, ExistsExpr):
            neg = 'NOT ' if n.negated else ''
            return f"{neg}EXISTS (...)"
        if isinstance(n, SubqueryExpr):
            return '(SELECT ...)'
        return str(n)
