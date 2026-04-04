from __future__ import annotations
"""AST expression → SQL text formatter (used for default column names)."""

from parser.ast import (
    AggregateCall, AliasExpr, BinaryExpr, ColumnRef, FunctionCall,
    IsNullExpr, Literal, StarExpr, UnaryExpr,
)
from parser.precedence import Precedence

# Precedence of binary operators for parenthesisation
_OP_PREC = {
    'OR': Precedence.OR, 'AND': Precedence.AND,
    '=': Precedence.COMPARISON, '!=': Precedence.COMPARISON,
    '<': Precedence.COMPARISON, '>': Precedence.COMPARISON,
    '<=': Precedence.COMPARISON, '>=': Precedence.COMPARISON,
    '||': Precedence.CONCAT,
    '+': Precedence.ADDITION, '-': Precedence.ADDITION,
    '*': Precedence.MULTIPLY, '/': Precedence.MULTIPLY, '%': Precedence.MULTIPLY,
}


class Formatter:
    """Converts AST expression nodes back into SQL text."""

    @staticmethod
    def expr_to_sql(expr: object) -> str:
        return Formatter._fmt(expr, 0)

    @staticmethod
    def _fmt(node: object, parent_prec: int) -> str:
        if isinstance(node, ColumnRef):
            if node.table:
                return f"{node.table}.{node.column}"
            return node.column

        if isinstance(node, Literal):
            if node.value is None:
                return 'NULL'
            if isinstance(node.value, bool):
                return 'TRUE' if node.value else 'FALSE'
            if isinstance(node.value, str):
                escaped = node.value.replace("'", "''")
                return f"'{escaped}'"
            return str(node.value)

        if isinstance(node, BinaryExpr):
            prec = _OP_PREC.get(node.op, 0)
            left_s = Formatter._fmt(node.left, prec)
            right_s = Formatter._fmt(node.right, prec)
            result = f"{left_s} {node.op} {right_s}"
            if prec < parent_prec:
                result = f"({result})"
            return result

        if isinstance(node, UnaryExpr):
            operand_s = Formatter._fmt(node.operand, Precedence.UNARY)
            if node.op == 'NOT':
                return f"NOT {operand_s}"
            return f"{node.op}{operand_s}"

        if isinstance(node, AggregateCall):
            if node.args and isinstance(node.args[0], StarExpr):
                return f"{node.name}(*)"
            args_s = ', '.join(Formatter._fmt(a, 0) for a in node.args)
            distinct_s = 'DISTINCT ' if node.distinct else ''
            return f"{node.name}({distinct_s}{args_s})"

        if isinstance(node, FunctionCall):
            args_s = ', '.join(Formatter._fmt(a, 0) for a in node.args)
            return f"{node.name}({args_s})"

        if isinstance(node, IsNullExpr):
            inner = Formatter._fmt(node.expr, 0)
            if node.negated:
                return f"{inner} IS NOT NULL"
            return f"{inner} IS NULL"

        if isinstance(node, AliasExpr):
            return Formatter._fmt(node.expr, parent_prec)

        if isinstance(node, StarExpr):
            if node.table:
                return f"{node.table}.*"
            return '*'

        return str(node)
