from __future__ import annotations
"""AST → SQL text formatter."""
from parser.ast import (AggregateCall, AliasExpr, BetweenExpr, BinaryExpr, CaseExpr,
                         CastExpr, ColumnRef, FunctionCall, InExpr, IsNullExpr,
                         LikeExpr, Literal, StarExpr, UnaryExpr)
from parser.precedence import Precedence

_OP_PREC = {'OR':1,'AND':2,'=':4,'!=':4,'<':4,'>':4,'<=':4,'>=':4,
            '||':5,'+':6,'-':6,'*':7,'/':7,'%':7}

class Formatter:
    @staticmethod
    def expr_to_sql(expr: object) -> str:
        return Formatter._fmt(expr, 0)

    @staticmethod
    def _fmt(n: object, pp: int) -> str:
        if isinstance(n, ColumnRef):
            return f"{n.table}.{n.column}" if n.table else n.column
        if isinstance(n, Literal):
            if n.value is None: return 'NULL'
            if isinstance(n.value, bool): return 'TRUE' if n.value else 'FALSE'
            if isinstance(n.value, str): return f"'{n.value}'"
            return str(n.value)
        if isinstance(n, BinaryExpr):
            p = _OP_PREC.get(n.op, 0)
            s = f"{Formatter._fmt(n.left, p)} {n.op} {Formatter._fmt(n.right, p)}"
            return f"({s})" if p < pp else s
        if isinstance(n, UnaryExpr):
            o = Formatter._fmt(n.operand, Precedence.UNARY)
            return f"NOT {o}" if n.op == 'NOT' else f"{n.op}{o}"
        if isinstance(n, AggregateCall):
            if n.args and isinstance(n.args[0], StarExpr): return f"{n.name}(*)"
            a = ', '.join(Formatter._fmt(x, 0) for x in n.args)
            d = 'DISTINCT ' if n.distinct else ''
            return f"{n.name}({d}{a})"
        if isinstance(n, FunctionCall):
            a = ', '.join(Formatter._fmt(x, 0) for x in n.args)
            return f"{n.name}({a})"
        if isinstance(n, IsNullExpr):
            i = Formatter._fmt(n.expr, 0)
            return f"{i} IS NOT NULL" if n.negated else f"{i} IS NULL"
        if isinstance(n, AliasExpr): return Formatter._fmt(n.expr, pp)
        if isinstance(n, StarExpr): return f"{n.table}.*" if n.table else '*'
        if isinstance(n, CaseExpr): return 'CASE ... END'
        if isinstance(n, CastExpr):
            return f"CAST({Formatter._fmt(n.expr, 0)} AS {n.type_name.name if n.type_name else '?'})"
        if isinstance(n, InExpr):
            vs = ', '.join(Formatter._fmt(v, 0) for v in n.values)
            neg = ' NOT' if n.negated else ''
            return f"{Formatter._fmt(n.expr, 0)}{neg} IN ({vs})"
        if isinstance(n, BetweenExpr):
            neg = ' NOT' if n.negated else ''
            return f"{Formatter._fmt(n.expr, 0)}{neg} BETWEEN {Formatter._fmt(n.low, 0)} AND {Formatter._fmt(n.high, 0)}"
        if isinstance(n, LikeExpr):
            neg = ' NOT' if n.negated else ''
            return f"{Formatter._fmt(n.expr, 0)}{neg} LIKE {Formatter._fmt(n.pattern, 0)}"
        return str(n)
