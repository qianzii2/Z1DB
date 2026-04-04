from __future__ import annotations
"""AST → SQL text."""
from parser.ast import *
from parser.precedence import Precedence
_P = {'OR':1,'AND':2,'=':4,'!=':4,'<':4,'>':4,'<=':4,'>=':4,'||':5,'+':6,'-':6,'*':7,'/':7,'%':7}

class Formatter:
    @staticmethod
    def expr_to_sql(e: object) -> str: return Formatter._f(e,0)
    @staticmethod
    def _f(n: object, pp: int) -> str:
        if isinstance(n, ColumnRef): return f"{n.table}.{n.column}" if n.table else n.column
        if isinstance(n, Literal):
            if n.value is None: return 'NULL'
            if isinstance(n.value, bool): return 'TRUE' if n.value else 'FALSE'
            if isinstance(n.value, str): return f"'{n.value}'"
            return str(n.value)
        if isinstance(n, BinaryExpr):
            p = _P.get(n.op,0); s = f"{Formatter._f(n.left,p)} {n.op} {Formatter._f(n.right,p)}"
            return f"({s})" if p < pp else s
        if isinstance(n, UnaryExpr):
            o = Formatter._f(n.operand, Precedence.UNARY)
            return f"NOT {o}" if n.op == 'NOT' else f"{n.op}{o}"
        if isinstance(n, AggregateCall):
            if n.args and isinstance(n.args[0], StarExpr): return f"{n.name}(*)"
            a = ', '.join(Formatter._f(x,0) for x in n.args)
            return f"{n.name}({'DISTINCT ' if n.distinct else ''}{a})"
        if isinstance(n, FunctionCall):
            return f"{n.name}({', '.join(Formatter._f(x,0) for x in n.args)})"
        if isinstance(n, WindowCall):
            fn = Formatter._f(n.func, 0)
            parts = []
            if n.partition_by: parts.append('PARTITION BY ' + ', '.join(Formatter._f(x,0) for x in n.partition_by))
            if n.order_by: parts.append('ORDER BY ' + ', '.join(Formatter._f(sk.expr,0) for sk in n.order_by))
            return f"{fn} OVER ({' '.join(parts)})"
        if isinstance(n, IsNullExpr):
            return f"{Formatter._f(n.expr,0)} IS {'NOT ' if n.negated else ''}NULL"
        if isinstance(n, AliasExpr): return Formatter._f(n.expr, pp)
        if isinstance(n, StarExpr): return f"{n.table}.*" if n.table else '*'
        if isinstance(n, CaseExpr): return 'CASE ... END'
        if isinstance(n, CastExpr):
            return f"CAST({Formatter._f(n.expr,0)} AS {n.type_name.name if n.type_name else '?'})"
        if isinstance(n, InExpr):
            return f"{Formatter._f(n.expr,0)}{' NOT' if n.negated else ''} IN (...)"
        if isinstance(n, BetweenExpr):
            return f"{Formatter._f(n.expr,0)}{' NOT' if n.negated else ''} BETWEEN ..."
        if isinstance(n, LikeExpr):
            return f"{Formatter._f(n.expr,0)}{' NOT' if n.negated else ''} LIKE ..."
        if isinstance(n, ExistsExpr): return f"{'NOT ' if n.negated else ''}EXISTS (...)"
        if isinstance(n, SubqueryExpr): return '(SELECT ...)'
        return str(n)
